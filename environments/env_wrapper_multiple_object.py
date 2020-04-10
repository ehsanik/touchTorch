# A wrapper that enables supporting multiple objects in one experiment
class MultipleObjectWrapper:

    def __init__(self, args, object_paths):
        self.list_of_envs = {}
        self.environment_type = args.environment
        self.object_paths = object_paths
        self.number_of_cp = args.number_of_cp
        self.force_h = args.force_h
        self.state_h = args.state_h
        self.qualitative_size = args.qualitative_size
        if args.render:
            assert len(object_paths) == 1, 'if gui only one object can be visualized'
        for obj in self.object_paths:
            self.list_of_envs[obj] = self.environment_type(args, self.object_paths[obj], obj)


    def reset(self):
        for obj in self.object_paths:
            self.list_of_envs[obj].reset()

    def init_location_and_apply_force(self, forces, initial_state, object_num=None, list_of_contact_points=None):
        assert list_of_contact_points is not None
        object_name = initial_state.object_name
        return self.list_of_envs[object_name].init_location_and_apply_force(forces, initial_state, object_num, list_of_contact_points)
