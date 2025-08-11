# src/util/urdf_to_dh/generate_dh.py
# This module generates Denavit-Hartenberg parameters from a URDF file.

from anytree import AnyNode, LevelOrderIter
import numpy as np
import pandas as pd
import math

import urdf_to_dh.kinematics_helpers as kh
import urdf_to_dh.geometry_helpers as gh
import urdf_to_dh.urdf_helpers as uh

class GenerateDhParams():
    def __init__(self, urdf_file: str):
        print(f'Initializing DH parameter generator...')
        self.urdf_joints = {}
        self.urdf_links = {}
        self.urdf_file = urdf_file
        self.urdf_tree_nodes = []
        self.root_link = None
        self.dh_dict = {}
        self.inertia_dict = {}
        self.verbose = False

        print('URDF file = %s' % self.urdf_file)

    def parse_urdf(self):
        # Get the root of the URDF and extract all of the joints
        urdf_root = uh.get_urdf_root(self.urdf_file)

        # Parse all links first and add to tree
        for child in urdf_root:
            if child.tag == 'link':
                _, link_data = uh.process_link(child)
                self.urdf_links[child.get('name')] = {'rel_tf': np.eye(4), 'abs_tf': np.eye(4), 'dh_tf': np.eye(4), 
                                                      'abs_dh_tf': np.eye(4), 'dh_found': False, 'mass': link_data['mass'],
                                                      'center_of_mass': link_data['center_of_mass'], 
                                                      'inertia_tensor': link_data['inertia_tensor']}
                node = AnyNode(id=child.get('name'), parent=None, children=None, type='link')
                self.urdf_tree_nodes.append(node)

        # Parse all joints and add to tree
        for child in urdf_root:
            if child.tag == 'joint':
                joint_name, joint_data = uh.process_joint(child)
                self.urdf_joints[joint_name] = joint_data
                node = AnyNode(id=joint_name, parent=None, children=None, type='joint')

                # Find parent and child link
                for n in self.urdf_tree_nodes:
                    if n.id == joint_data['parent']:
                        node.parent = n
                    if n.id == joint_data['child']:
                        n.parent = node
                self.urdf_tree_nodes.append(node)

        # Find root link
        num_nodes_no_parent = 0
        for n in self.urdf_tree_nodes:
            if n.parent == None:
                num_nodes_no_parent += 1
                self.root_link = n

        if num_nodes_no_parent == 1:
            # Root link DH will be identity, set dh_found = True
            # TO DO: Probably not needed since order iter is used
            self.urdf_links[self.root_link.id]['dh_found'] = True
            # print("URDF Tree:")
            # for pre, _, node in RenderTree(self.root_link):
            #     print('%s%s' % (pre, node.id))

            # print("Joint Info:")
            # pprint.pprint(self.urdf_joints)
        else:
            raise RuntimeError("Error: Should only be one root link")
        
    def calculate_tfs_in_world_frame(self):
        # print("Calculate world tfs:")
        for n in LevelOrderIter(self.root_link):
            if n.type == 'link' and n.parent != None:
                # print("\nget tf from ", n.parent.parent.id, " to ", n.id)
                
                parent_tf_world = self.urdf_links[n.parent.parent.id]['abs_tf']
                xyz = self.urdf_joints[n.parent.id]['xyz']
                rpy = self.urdf_joints[n.parent.id]['rpy']
                tf = np.eye(4)
                tf[0:3, 0:3] = kh.get_extrinsic_rotation(rpy)
                tf[0:3, 3] = xyz
                self.urdf_links[n.id]['rel_tf'] = tf

                abs_tf = np.eye(4)
                abs_tf = np.matmul(parent_tf_world, tf)
                self.urdf_links[n.id]['abs_tf'] = abs_tf
    
    def calculate_params(self):
        robot_dh_params = []
        robot_mass_params = []

        for urdf_node in LevelOrderIter(self.root_link):
            if urdf_node.type == 'link' and self.urdf_links[urdf_node.id]['dh_found'] == False:
                # print("\n\nprocess dh params for ", urdf_node.id)

                # TF from current link frame to world frame
                link_to_world = self.urdf_links[urdf_node.id]['abs_tf']

                # DH frame from parent link frame to world frame
                parent_to_world_dh = self.urdf_links[urdf_node.parent.parent.id]['abs_dh_tf']

                # TF from link frame to parent dh frame
                link_to_parent_dh = np.matmul(kh.inv_tf(parent_to_world_dh), link_to_world)

                # Find DH parameters
                axis = np.matmul(link_to_parent_dh[0:3, 0:3], self.urdf_joints[urdf_node.parent.id]['axis'])
                
                dh_params = self.__get_joint_dh_params(link_to_parent_dh, axis)

                dh_frame = kh.get_dh_frame(dh_params)
                abs_dh_frame = np.matmul(parent_to_world_dh, dh_frame)

                self.urdf_links[urdf_node.id]['dh_tf'] = dh_frame

                self.urdf_links[urdf_node.id]['abs_dh_tf'] = abs_dh_frame
                robot_dh_params.append([urdf_node.parent.id, urdf_node.parent.parent.id, urdf_node.id] + list(dh_params.round(5)))
                robot_mass_params.append([urdf_node.id, self.urdf_links[urdf_node.id]['mass'],
                                          self.urdf_links[urdf_node.id]['center_of_mass'], self.urdf_links[urdf_node.id]['inertia_tensor']])


        dh_pd_frame = pd.DataFrame(robot_dh_params, columns=['joint', 'parent', 'child', 'd', 'theta', 'r', 'alpha'])
        dh_pd_frame['theta'] = dh_pd_frame['theta'] * 180.0 / math.pi
        dh_pd_frame['alpha'] = dh_pd_frame['alpha'] * 180.0 / math.pi

        inertia_pd_frame = pd.DataFrame(robot_mass_params, columns=['link', 'mass', 'com', 'inertia_tensor'])

        self.dh_dict = dh_pd_frame.to_dict()
        self.inertia_dict = inertia_pd_frame.to_dict()

    def __get_joint_dh_params(self, rel_link_frame: np.ndarray, axis: np.ndarray):
        dh_params = np.zeros(4)
        origin_xyz = rel_link_frame[0:3, 3]
        z_axis = np.array([0, 0, 1])

        # Collinear case
        if gh.are_collinear(np.zeros(3), z_axis, origin_xyz, axis):
            dh_params = self.__process_collinear_case(origin_xyz, rel_link_frame[0:3, 0])

        # Parallel case
        elif gh.are_parallel(z_axis, axis):
            dh_params = self.__process_parallel_case(origin_xyz)

        # Intersect case
        elif gh.lines_intersect(np.zeros(3), z_axis, origin_xyz, axis)[0]:
            dh_params = self.__process_intersection_case(origin_xyz, axis)

        # Skew case
        else:
            dh_params = self.__process_skew_case(origin_xyz, axis)


        for i in range(len(dh_params)):
            if np.isnan(dh_params[i]):
                dh_params[i] = 0
        return dh_params
    
    def __process_collinear_case(self, origin, xaxis):
        dh_params = np.zeros(4)
        dh_params[0] = origin[2]
        return dh_params

    def __process_parallel_case(self, origin):
        dh_params = np.zeros(4)
        dh_params[0] = origin[2]
        dh_params[1] = math.atan2(origin[1], origin[0])
        dh_params[2] = math.sqrt(origin[0]**2 + origin[1]**2)
        return dh_params

    def __process_intersection_case(self, origin, axis):
        dh_params = np.zeros(4)
        _, x = gh.lines_intersect(np.zeros(3), np.array([0, 0, 1]), origin, axis)
        dh_params[0] = float(x[1, 0])

        zaxis = np.array([0., 0., 1.])
        xaxis = np.array([1., 0., 0.])

        for i in range(0,3):
            if abs(axis[i]) < 1.e-5:
                axis[i] = 0

        cn = np.cross(zaxis, axis)
        for i in range(0,3):
            if abs(cn[i]) < 1.e-6:
                cn[i] = 0
        if (cn[0] < 0):
            cn = cn * -1
        dh_params[1] = math.atan2(cn[1], cn[0])
        # print(math.atan2(np.dot(np.cross(xaxis, cn), zaxis), np.dot(xaxis, cn)))

        dh_params[2] = 0

        vn = cn / np.linalg.norm(cn)
        dh_params[3] = math.atan2(np.dot(np.cross(zaxis, axis), vn), np.dot(zaxis, axis))

        return dh_params

    def __process_skew_case(self, origin, direction):
        pointA = np.zeros(3)
        pointB = np.zeros(3)
        dh_params = np.zeros(4)

        # Find closest points along parent z-axis (pointA) and joint axis (pointB)
        t = -1.0 * (origin[0] * direction[0] + origin[1] * direction[1]) / (direction[0]**2 + direction[1]**2)
        pointB = origin + t * direction
        pointA[2] = pointB[2]

        # 'd' is offset along parent z axis
        dh_params[0] = pointA[2]

        # 'r' is the length of the common normal
        dh_params[2] = np.linalg.norm(pointB - pointA)

        # 'theta' is the angle between the x-axis and the common normal
        dh_params[1] = math.atan2(pointB[1], pointB[0])

        # 'alpha' is the angle between the current z-axis and the joint axis
        cn = pointB - pointA
        vn = cn / np.linalg.norm(cn)
        zaxis = np.array([0, 0, 1])
        dh_params[3] = math.atan2(np.dot(np.cross(zaxis, direction), vn), np.dot(zaxis, direction))

        return dh_params