import bpy
import enum
import tempfile
from contextlib import contextmanager

import skimage
import skimage.io

import os

from toolbox.io.images import load_hdr

BPY_VERSION_MAJOR = bpy.app.version[0]
BPY_VERSION_MINOR = bpy.app.version[1]

IS_BPY_279 = BPY_VERSION_MAJOR == 2 and BPY_VERSION_MINOR < 80

class BackgroundMode(enum.Enum):
    DISABLED = enum.auto()
    ENVMAP = enum.auto()
    COLOR = enum.auto()

class Engine(enum.Enum):
    CYCLES = 'CYCLES'
    # BLENDER = 'BLENDER_RENDER'
    EEVEE = 'BLENDER_EEVEE'
    WORKBENCH = 'BLENDER_WORKBENCH'

class Scene:
    _current = None
    _context_stack = []

    @staticmethod
    def current():
        return Scene._current

    def __init__(self, app, shape, device='GPU',
                 engine=Engine.CYCLES,
                 tile_size=(40, 40),
                 aa_samples=196,
                 diffuse_samples=2,
                 specular_samples=2,
                 num_samples=256,
                 background_mode = BackgroundMode.ENVMAP,
                 background_color=(0, 0, 0, 1),
                 bscene=None,
                 improved_quality = False,
                 gpu_indices = None):
        if not app.initialized:
            raise RuntimeError('App must be initialized.')

        self._app = app
        self.bpy_v_major = bpy.app.version[0]
        self.bpy_v_minor = bpy.app.version[1]

        if bscene is None:
            self.bobj = bpy.context.scene
        else:
            self.bobj = bscene

        self.bobj.render.engine = engine.value
        self.bobj.cycles.device = device
        self.bobj.cycles.samples = num_samples
        self.bobj.render.tile_x = tile_size[0]
        self.bobj.render.tile_y = tile_size[1]
        self.bobj.render.resolution_percentage = 100
        
        # Set World Ambient color to Black
        if improved_quality:
            self.bobj.world.use_nodes = True
            self.bobj.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
            self.bobj.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0

            self.bobj.view_settings.view_transform = "Filmic"
            self.bobj.view_settings.look = "High Contrast"
        
        # self.bobj.cycles.samples = samples
        # self.bobj.cycles.diffuse_samples = diffuse_samples
        # self.bobj.cycles.glossy_samples = specular_samples
        # self.bobj.cycles.aa_samples = aa_samples
        # self.bobj.cycles.progressive = 'BRANCHED_PATH'
        # self.bobj.cycles.caustics_refractive = False
        # self.bobj.cycles.caustics_reflective = False
        # self.bobj.cycles.min_bounces = 2
        # self.bobj.cycles.max_bounces = 6

        self.background_mode = background_mode
        self.background_color = background_color

        self.bobj.use_nodes = True
        
        if improved_quality:
            # Adjust Ambient Occlusion Settings
            self.bobj.world.light_settings.use_ambient_occlusion = False # Yes, False!
            self.bobj.world.light_settings.distance = 0.2
            self.bobj.view_layers['View Layer'].use_pass_ambient_occlusion = True

            if engine == Engine.CYCLES:
                # Activate Denoising Data
                self.bobj.view_layers['View Layer'].cycles.denoising_store_passes = True
                
                # Add Denoise Node to Compositing
                tree = self.bobj.node_tree
                tree_nodes = tree.nodes
                tree_links = tree.links
                
                for link in tree_links:
                    tree_links.remove(link)

                composite_node = tree.nodes["Composite"] # 'CompositorNodeComposite'
                render_layers = tree.nodes["Render Layers"] # 'CompositorNodeRLayers'
                denoise_node = tree_nodes.new(type = "CompositorNodeDenoise")
                denoise_node.use_hdr = True
                
                mix_node = tree_nodes.new(type = "CompositorNodeMixRGB")
                mix_node.blend_type = "MULTIPLY"
                mix_node.inputs["Fac"].default_value = 0.4
                
                tree_links.new(render_layers.outputs["Noisy Image"], mix_node.inputs[1])
                tree_links.new(render_layers.outputs["AO"], mix_node.inputs[2])

                # tree_links.new(render_layers.outputs["Noisy Image"], denoise_node.inputs["Image"])
                tree_links.new(mix_node.outputs[0], denoise_node.inputs["Image"])
                tree_links.new(render_layers.outputs["Denoising Normal"], denoise_node.inputs["Normal"])
                tree_links.new(render_layers.outputs["Denoising Albedo"], denoise_node.inputs["Albedo"])
                tree_links.new(denoise_node.outputs["Image"], composite_node.inputs["Image"])

        if engine == Engine.CYCLES:
            
            # Make envmap invisible to camera.
            if self.background_mode == BackgroundMode.DISABLED:
                self.bobj.world.cycles_visibility.camera = False
            else:
                self.bobj.world.cycles_visibility.camera = True

            if device == 'GPU':
                if self.bpy_v_major == 2 and self.bpy_v_minor < 80:
                    prefs = bpy.context.user_preferences.addons['cycles'].preferences
                else:
                    prefs = bpy.context.preferences.addons['cycles'].preferences
                
                bpy.context.scene.cycles.device = "GPU"
                prefs.compute_device_type = 'CUDA'
                deviceList = prefs.get_devices()
                
                # for deviceTuple in deviceList:
                #     print("Devices:")
                #     for device in deviceTuple:
                #         print(f"\t{device.name} ({device.type}) {device.use}")
                
                for d_idx, d in enumerate(prefs.devices):
                    # d["use"] = 1 # Using all devices, include GPU and CPU

                    if d.type == 'CUDA':
                        d.use = True
                        
                        if gpu_indices is not None:
                            if d_idx in gpu_indices: # >= 2: # 3: # Only use GPU 2 and 3
                                d.use = True
                            else:
                                d.use = False
                            
                        print(d["name"], d_idx, d["use"])
                        
                # prefs.devices[0].use = True
                # bpy.ops.wm.save_userpref()

        self.camera = None
        self.shape = shape
        self.meshes = []
        self.bmats = []

        self._envmap_mapping_node = None

    def add_bmat(self, bmat):
        self.bmats.append(bmat)

    def clear_bmats(self):
        while len(self.bmats) > 0:
            bmat = self.bmats.pop()
            bmat.bobj.name = bmat.bobj.name
            del bmat

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self.bobj.render.resolution_y = shape[0]
        self.bobj.render.resolution_x = shape[1]

    def clear(self):
        for obj in self.bobj.objects:
            self.bobj.objects.unlink(obj)

    def set_active_camera(self, camera):
        self.camera = camera
        self.bobj.camera = camera.bobj

    @property
    def active_camera(self):
        return self.camera

    def render(self, path):
        if self.active_camera is None:
            raise RuntimeError('No active camera.')

        if str(path).endswith('.hdr'):
            bpy.context.scene.render.image_settings.file_format = 'HDR'
        elif str(path).endswith('.exr'):
            bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        else:
            bpy.context.scene.render.image_settings.file_format = 'PNG'

        self.bobj.render.filepath = str(path)
        bpy.ops.render.render(write_still=True, scene=self.bobj.name)

    def render_to_array(self, format='png'):
        """
        Renders the image to an array. If the format is 'hdr' or 'exr' the
        rendering is a linear HDR image. If format is 'png' then the image
        will be processed with Blender's default tonemapping and postprocessing
        effects.

        :param format: one of png, exr, or hdr.
        :return: rendered image
        """
        with tempfile.NamedTemporaryFile(suffix=f'.{format}') as f:
            self.render(f.name)
            if format in {'hdr', 'exr'}:
                os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
                return load_hdr(f.name)
            else:
                return skimage.img_as_float(skimage.io.imread(f.name))

    def clear_envmap(self):
        scene = self.bobj
        scene.world.use_nodes = True
        scene.world.node_tree.nodes.clear()
        scene.world.node_tree.links.clear()

    def set_envmap(self, path, scale=1.0, rotation=(0.0, 0.0, 0.0)):
        print(f"Setting envmap to {path}")

        self.clear_envmap()

        scene = self.bobj
        scene.world.use_nodes = True
        nodes = scene.world.node_tree.nodes
        links = scene.world.node_tree.links

        lightpath_node = nodes.new(type='ShaderNodeLightPath')

        texcoord_node = nodes.new(type='ShaderNodeTexCoord')
        mapping_node = nodes.new(type='ShaderNodeMapping')
        
        if IS_BPY_279:
            mapping_node.rotation = rotation
        else:
            mapping_node.inputs['Rotation'].default_value = rotation
            
        self._envmap_mapping_node = mapping_node

        env_image = bpy.data.images.load(filepath=str(path))
        envtex_node = nodes.new(type="ShaderNodeTexEnvironment")
        envtex_node.image = env_image
        env_bg_node = nodes.new(type="ShaderNodeBackground")
        env_bg_node.inputs[1].default_value = scale

        if self.background_mode == BackgroundMode.COLOR:
            white_bg_node = nodes.new(type='ShaderNodeBackground')
            white_bg_node.inputs[0].default_value = self.background_color
            bg_select_node = nodes.new(type='ShaderNodeMixShader')
            links.new(bg_select_node.inputs[0], lightpath_node.outputs[0])
            links.new(bg_select_node.inputs[1], env_bg_node.outputs[0])
            links.new(bg_select_node.inputs[2], white_bg_node.outputs[0])
            bg_output = bg_select_node.outputs[0]
        else:
            bg_output = env_bg_node.outputs[0]

        output_node = nodes.new(type="ShaderNodeOutputWorld")

        links.new(mapping_node.inputs[0], texcoord_node.outputs[0])
        links.new(envtex_node.inputs[0], mapping_node.outputs[0])
        links.new(env_bg_node.inputs[0], envtex_node.outputs[0])
        links.new(output_node.inputs[0], bg_output)

    def set_envmap_rotation(self, rotation):
        if self._envmap_mapping_node is None:
            raise ValueError('Envmap has not been set yet.')
        
        if IS_BPY_279:
            self._envmap_mapping_node.rotation = rotation
        else:
            self._envmap_mapping_node.inputs['Rotation'].default_value = rotation

    def set_render_engine(self, engine_enum):
        self.bobj.render.engine = engine_enum.value

    def clear_materials(self):
        for material in bpy.data.materials:
            if material.name == 'material_floor':
                continue
            print(f'Removing material {material.name!r}')
            # material.user_clear()
            bpy.data.materials.remove(material, do_unlink=True)

    def clear_meshes(self):
        for obj in self.bobj.objects:
            if obj.name == 'floor':
                continue
            if obj.type == 'MESH':
                print('Removing mesh {obj.name!r}')
                bpy.data.objects.remove(obj, do_unlink=True)

    @contextmanager
    def select(self):
        if Scene._current is not None:
            raise RuntimeError('Only one scene may be active.')

        Scene._current = self
        
        if self.bpy_v_major == 2 and self.bpy_v_minor < 80:
            scene = bpy.context.screen.scene
        else:
            scene = bpy.context.scene
        
        Scene._context_stack.append(scene) # bpy.context.screen.scene
        scene = self.bobj # bpy.context.screen.scene
        try:
            yield self
        finally:
            Scene._current = None
            scene = Scene._context_stack.pop() # bpy.context.screen.scene
