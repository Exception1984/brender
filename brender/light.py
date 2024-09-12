import bpy

class Light():
    _next_id = 0
    _current = None
    _context_stack = []
    
    def __init__(self, obj, name=None):
        self.bobj = obj
        self.bobjs = [self.bobj]
        
        if name is None:
            name = f'brender_light_{self._next_id}'
            Light._next_id += 1
            self.bobj.name = name


class PointLight(Light):
    def __init__(self, size = 1.5, location = (0.0, 0.0, 0.0), energy = 500, **kwargs):
        # scene = bpy.context.scene
        
        # bpy.ops.object.light_add(type='POINT', radius=1.0, align='WORLD', location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.0, 0.0, 0.0))
        bpy.ops.object.light_add(type='POINT', location=location)
        bobj = bpy.context.selected_objects[0]
        
        bobj.data.shadow_soft_size = size
        bobj.data.energy = energy
        
        # # Create new lamp datablock
        # lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')

        # # Create new object with our lamp datablock
        # lamp_object = bpy.data.objects.new(name="New Lamp", object_data = lamp_data)

        # # Link lamp object to the scene so it'll appear in this scene
        # scene.objects.link(lamp_object)

        # # Place lamp to a specified location
        # lamp_object.location = (5.0, 5.0, 5.0)

        # # And finally select it make active
        # lamp_object.select = True
        # scene.objects.active = lamp_object
        
        super().__init__(bobj, **kwargs)

class SunLight(Light):
    def __init__(self, size = 1.5, location = (0.0, 0.0, 0.0), energy = 500, **kwargs):
        # scene = bpy.context.scene
        
        # bpy.ops.object.light_add(type='POINT', radius=1.0, align='WORLD', location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.0, 0.0, 0.0))
        bpy.ops.object.light_add(type = 'SUN', location=location)
        bobj = bpy.context.selected_objects[0]
        
        bobj.data.shadow_soft_size = size
        bobj.data.energy = energy
        
        super().__init__(bobj, **kwargs)