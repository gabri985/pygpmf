from scipy.spatial.transform import Rotation as R
from collections import namedtuple
import numpy as np

from . import parse

CoriData = namedtuple('CoriData', [
    'cts',
    'z',
    'y',
    'x'
])

IoriData = namedtuple('IoriData', [
    'cts',
    'z',
    'y',
    'x'
])


def extract_blocks(stream, fourcc):
    """ Extract data blocks from binary stream for a given foorcc code.

    This is a generator on lists `KVLItem` objects. In
    the GPMF stream, data comes into blocks of several
    different data items. For each of these blocks we return a list.

    Parameters
    ----------
    stream: bytes
        The raw GPMF binary stream
    fourcc: string
        The four charakter key that marks a segement of data in GPMF.

    Returns
    -------
    items_generator: generator
        Generator of lists of `KVLItem` objects
    """
    for s in parse.filter_klv(stream, 'STRM'):
        content = []
        is_cc = False
        for elt in s.value:
            content.append(elt)            
            if elt.key == fourcc:
                is_cc = True
        if is_cc:
            yield content


def parse_iori_block(block):
    """Turn IORI data blocks into `IoriData` objects.
    Convert rotation from Quaternion format to Euler angles.

    Parameters
    ----------
    block: list of KVLItem
        A list of KVLItem corresponding to a IORI data block.

    Returns
    -------
    iori_data: IoriData
        A IoriData object holding the IORI information of a block.
    """
    block_dict = {
        s.key: s for s in block
    }
    data = block_dict['IORI'].value * 1.0 / block_dict["SCAL"].value

    rotation = np.array([R.from_quat(q).as_euler('zyx', degrees=True) for q in data])
    z, y, x = rotation.T

    return IoriData(
        cts = block_dict['STMP'].value,
        z = z,
        y = y,
        x = x,
    )


def parse_cori_block(block):
    """Turn CORI data blocks into `CoriData` objects.
    Convert rotation from Quaternion format to Euler angles.

    Parameters
    ----------
    block: list of KVLItem
        A list of KVLItem corresponding to a CORI data block.

    Returns
    -------
    cori_data: CoriData
        A CoriData object holding the CORI information of a block.
    """
    block_dict = {
        s.key: s for s in block
    }
    data = block_dict['CORI'].value * 1.0 / block_dict["SCAL"].value

    rotation = np.array([R.from_quat(q).as_euler('zyx', degrees=True) for q in data])
    z, y, x = rotation.T

    return CoriData(
        cts = block_dict['STMP'].value,
        z = z,
        y = y,
        x = x,
    )