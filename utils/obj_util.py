import numpy as np
import pdb


def obtain_all_vertices_from_obj(obj_path_name):
    with open(obj_path_name) as f:
        lines = [l.replace('\n', '') for l in f]
    set_of_vertices = []
    set_of_faces = []
    for l in lines:
        splitted = l.split(' ')
        if splitted[0] == 'v':
            vertex = [float(x) for x in splitted[1:] if x != '']
            set_of_vertices.append(vertex)
        if splitted[0] == 'f':
            face = [int(x) - 1 for x in splitted[1:] if x != '']
            set_of_faces.append(face)
    set_of_vertices = np.array(set_of_vertices)
    set_of_faces = np.array(set_of_faces)
    set_of_middle_points = [set_of_vertices[f].mean(axis=0) for f in set_of_faces]
    set_of_middle_points = np.array(set_of_middle_points)
    assert len(set_of_vertices) > 0, 'no vertices found {}'.format(obj_path_name)
    return set_of_vertices, set_of_faces, set_of_middle_points


def obtain_vertices_from_any_kind_obj(obj_path_name):
    import pywavefront
    scene = pywavefront.Wavefront(obj_path_name, strict=True, encoding="iso-8859-1", parse=False)
    scene.parse()
    pdb.set_trace()
