import bpy

class Brender:
    _instance = None

    def __init__(self):
        if Brender._instance:
            return Brender._instance
            # raise RuntimeError('There can only be one app instance.')
        else:
            self.initialized = False
            Brender._instance = self
        
    def deleteAllObjects(self):
        """
        Deletes all objects in the current scene
        """
        deleteListObjects = ['MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'HAIR', 'POINTCLOUD', 'VOLUME', 'GPENCIL',
                        'ARMATURE', 'LATTICE', 'EMPTY', 'LIGHT', 'LIGHT_PROBE', 'CAMERA', 'SPEAKER']

        # Select all objects in the scene to be deleted:

        for o in bpy.context.scene.objects:
            for i in deleteListObjects:
                if o.type == i:
                    o.select_set(False)
                else:
                    o.select_set(True)
        # Deletes all selected objects in the scene:

        bpy.ops.object.delete()

    def init(self, do_reset=True):
        # Import default scene and settings.
        bpy.ops.wm.read_factory_settings()

        if do_reset:
            self.deleteAllObjects()
            
            # bpy_version = bpy.app.version
            # if bpy_version[0] == 2 and bpy_version[1] > 79:
            #     lights = bpy.data.lights
            # else:
            #     lights = bpy.data.lamps,
            
            # # Clear data.
            
            # for bpy_data_iter in (
            #         bpy.data.objects,
            #         bpy.data.meshes,
            #         lights,
            #         bpy.data.cameras,
            #         bpy.data.materials,
            # ):
            #     for id_data in bpy_data_iter:
            #         bpy_data_iter.remove(id_data, do_unlink=True) # bpy.data.objects.remove(object, do_unlink=True)

        self.initialized = True