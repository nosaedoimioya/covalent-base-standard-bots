# urdf_helpers.py
# Author: Nosa Edoimioya
# Description: General helper functions to generate DH parameters based on robot urdf file.
# Version: 0.1
# Date: 03-27-2024
# Based on https://github.com/mcevoyandy/urdf_to_dh

import xml.etree.ElementTree as ET
import numpy as np
from typing import Tuple, Dict

# Helper functions for parsing the URDF
def get_urdf_root(urdf_file: str) -> ET.Element:
    """Parse a URDF for joints.

    Args:
        urdf_path (string): The absolute path to the URDF to be analyzed.

    Returns:
        root (xml object): root node of the URDF.
    """
    try:
        tree = ET.parse(urdf_file)
    except ET.ParseError:
        print('ERROR: Could not parse urdf file.')

    return tree.getroot()

def process_joint(joint: ET.Element) -> Tuple[str, Dict]:
    """Extracts the relevant joint info into a dictionary.
    Args: 
        joint Element (xml object): joint node of URDF
    Returns:
        joint_name (string): name of the joint
        joint_data (dict): dictionary of joint data
    """
    axis = np.array([1, 0, 0])
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    parent_link = ''
    child_link = ''

    joint_name = joint.get('name')

    for child in joint:
        if child.tag == 'axis':
            axis = np.array(child.get('xyz').split(), dtype=float)
        elif child.tag == 'origin':
            xyz = np.array(child.get('xyz').split(), dtype=float)
            rpy = np.array(child.get('rpy').split(), dtype=float)
        elif child.tag == 'parent':
            parent_link = child.get('link')
        elif child.tag == 'child':
            child_link = child.get('link')
    return joint_name, {'axis': axis, 'xyz': xyz, 'rpy': rpy, 'parent': parent_link, 'child': child_link, 'dh': np.zeros(4)}

def process_link(link: ET.Element) -> Tuple[str, Dict]:
    """Extracts the relevant link prp into a dictionary.
    Args: 
        link Element (xml object): link node of URDF
    Returns:
        link_name (string): name of the link
        link_data (dict): dictionary of link properties
    """
    mass = 0.0
    center_of_mass = np.zeros(3)
    inertia_tensor = np.zeros((3,3))

    link_name = link.get('name')

    for child in link:
        if child.tag == 'inertial':
            for grand_child in child:
                if grand_child.tag == 'mass':
                    mass = float(grand_child.get('value'))
                elif grand_child.tag == 'inertia':
                    inertia_tensor = np.array([
                                        [grand_child.get('ixx'),grand_child.get('ixy'),grand_child.get('ixz')],
                                        [grand_child.get('ixy'),grand_child.get('iyy'),grand_child.get('iyz')],
                                        [grand_child.get('ixz'),grand_child.get('iyz'),grand_child.get('izz')]
                                        ], dtype=float)
                elif grand_child.tag == 'origin':
                    center_of_mass = np.array(grand_child.get('xyz').split(), dtype=float)
            
            # Break loop after finding the 'inertial' tag    
            break

    return link_name, {'mass': mass, 'center_of_mass': center_of_mass, 'inertia_tensor': inertia_tensor}
            


