# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Functions to fit ExCAM beam shifts to actuator movements.

Functions to compute the matrix that transforms lateral beam shifts at
ExCAM into actuator commands (such as for FSM and OTA TTF).
"""
import numbers
import numpy as np
import os
from astropy.io import fits
import scipy.linalg as linalg
from numpy.linalg import LinAlgError

from cal.util import check, fit_shapes, loadyaml
from cal.util.insertinto import insertinto
from cal.psffit.psffit import psffit
import cal.pupilfit.pupilfit_open as pfo
import cal.lowfsfit.lowfsfit as lowfsfit

PFO_PATH = os.path.dirname(os.path.abspath(pfo.__file__))
LOWFSFIT_PATH = os.path.dirname(os.path.abspath(lowfsfit.__file__))
LOWFS_CONFIG_DEFAULT_FN = os.path.join(LOWFSFIT_PATH, 'config_files', 'lowfsfit_parms.yaml')

class CtransAbsFwd:
    """
    class for forward absolute transform object (EXCAM pixel to actuator command)
    overloads matrix multiply to properly include translation in the operation:
    U = tMat @ XY
    """
    def __init__(self, transAbsFwd):
        self.tMat = transAbsFwd

    def __matmul__(self, XY):
        """
        input:
           XY : numpy ndarray, for example, XY = np.array(xyOffsetList),
                where xyOffsetList = list of [x, y] = [[x1, y1], [x2, y2], ..., [xN, yN]]
                Each row of XY is an [x, y] duple.
                XY.shape = (N, 2)
                or vector of one x,y
                XY.shape = (2,)

        returns:
           U : numpy ndarray, np.array(commandList), where
               commandList = list of [u1, ..., uM] command sets
               Each row of U is a command of M actuators
               U.shape = (N, M)

        """

        # input XY must be ndarray
        if not isinstance(XY, np.ndarray):
            raise TypeError('input XY must be ndarray')

        # XY can have only real values
        check.real_array(XY, 'XY', TypeError)

        # if XY is 1-d of length 2, make it 2-d shape (1,2)
        if len(XY.shape) == 1 and XY.shape[0] == 2:
            XY = XY[np.newaxis, :]

        # XY must be 2-d
        check.twoD_array(XY, 'XY', TypeError)

        # each x,y is a duple
        N, n2 = XY.shape
        if n2 != 2:
            raise TypeError('XY must be list of x,y pairs')

        # transpose and add row of N ones
        XYtmp = np.vstack((XY.T, np.ones((N,))))

        # do the transform
        U1 = self.tMat @ XYtmp

        # last row of U1 should be ones
        U = U1[:-1, :]

        # return list of command sets
        return U.T

class CtransAbsRev:
    """
    class for reverse absolute transform object (actuator command to EXCAM pixel)
    overloads matrix multiply to properly include translation in the operation:
    XY = tMat @ U
    """
    def __init__(self, transAbsRev):
        self.tMat = transAbsRev
        self.Mact = transAbsRev.shape[1] - 1 # number actuators

    def __matmul__(self, U):
        """
        input:
           U : numpy ndarray, np.array(commandList), where
               commandList = list of [u1, ..., uM] command sets
               Each row of U is a command of M actuators
               U.shape = (N, M)
               or U is a vector of single command, length M actuators
               U.shape = (M,)

        returns:
           XY : numpy ndarray, for example, XY = np.array(xyOffsetList),
                where xyOffsetList = list of [x, y] = [[x1, y1], [x2, y2], ..., [xN, yN]]
                Each row of XY is an [x, y] duple.
                XY.shape = (N, 2)

        """

        # input U must be ndarray
        if not isinstance(U, np.ndarray):
            raise TypeError('input U must be ndarray')

        # U can have only real values
        check.real_array(U, 'U', TypeError)

        # if U is 1-d of length M, make it 2-d shape (1,M)
        if len(U.shape) == 1 and U.shape[0] == self.Mact:
            U = U[np.newaxis, :]

        # must be 1-d array of shape(
        check.twoD_array(U, 'U', TypeError)

        # each row must be command of M actuators
        N, M = U.shape
        if M != self.Mact:
            raise TypeError('U must be list of u vectors of length M actuators')

        # transpose and add row of ones
        U1 = np.vstack((U.T, np.ones((N,))))

        # do the transform
        XY = self.tMat @ U1

        # transpose XY to list of [x,y], shape = (N, 2)
        return XY.T

def calc_trans_matrix(xyOffsetList, commandList):
    """
    Compute the matrix that transforms star pixel offsets on EXCAM into fold
    mirror commands.

    Calibrates motions for actuators that move the beam (e.g., for
    the fast steering mirror (FSM), optical telescope assembly (OTA) pointing,
    or  OTA tip/tilt/fold (TTF) mirror) or move the coronagraph masks (e.g.,
    for the SPAM and LSAM).

    If [x y] is an EXCAM pixel, and [u1, ..., uM] is a command
    for M actuators, then the transform from actuators to pixel is:

    [ax1, ..., axM, tx   @  [u1, ..., uM, 1].T  =  [x,
     ay1, ..., ayM, ty]                             y]

    The transform matrix,

    A = [ax1, ..., axM, tx
         ay1, ..., ayM, ty]

    where tx, ty are translations in x, y pixels, is returned as transAbsRev.
    To utilize the overloaded matrix multiply operator @, first convert the
    array to the custom object with this command:
    transAbsRev = mf.CtransAbsRev(transAbsRev)

    The forward transform from EXCAM pixel to an actuator command set is:

    B = [b11, b12, b13
         ...
         bM1, bM2, bM3
         0,   0,   1  ]

    such that
    [u1, ..., uM, 1].T  =  B  @  [x, y, 1].T

    B is returned as transAbsFwd.
    To utilize the overloaded matrix multiply operator @, first convert the
    array to the custom object with this command:
    transAbsFwd = CtransAbsFwd(transAbsFwd)

    The forward and reverse absolute transforms are calculated from a given
    list of commands and corresponding EXCAM pixels. If the given list includes
    at least three sets of commands and pixels, then the transforms are
    calculated using using pseudo-inverse. The resulting transforms are affine
    transforms, including effects of rotation, scale, translation, and shear.
    Scale can be different in the x and y directions. Shear means a rectangle
    can map to a parallelogram. (see Matlab help for fitgeotrans.)

    For the special case of exactly two actuators and two given control points,
    there are not enough degrees of freedom for an affine transform.
    We can calculate the similarty transform (rotation, scale, and translation),
    a = d = s*cos(th), b = -c = s*sin(th)
    thus four unknows: s, th, tx, ty
    The four equations are derived from:
    [u1 u2  = T' * [x1 x2
     v1 v2          y1 y2
     1  1 ]         1  1 ]

    The four equations can be explicit, for example using the symbolic toolbox, or:
    https://math.stackexchange.com/questions/4010538/solve-for-2d-translation-rotation-and-scale-given-two-touch-point-movements

    The affine and similarity transforms calculated from the input lists of commands
    and EXCAM pixels are absolute transforms in that they include translation.
    However, the requirement, D-1050945, is for "...motions along the EXCAM pixel
    grid to motions of the FSM actuators which remove FSM piston." That is, the
    transform from differential dx, dy to differential actuator movement.

    But,the absolute transform which includes translation is not a linear
    operator. That is,
    transAbsFwd @ (x+dx, y+dy) != transAbsFwd @ (x, y) + transAbsFwd @ (dx, dy)
    because translation would be double counted. Instead, we extract the
    relative transform, transRelFwd, such that,
    transAbsFwd @ (x+dx, y+dy) == transAbsFwd @ (x, y) + transRelFwd @ (dx, dy)

    Similarly, we return a relative transform for the reverse direction,
    transAbsRev @ (u + du) == transAbsRev @ u + transRelRev @ du
    where u = (u1, ..., uM)

    The requirement is satisfied by the term, transRelFwd @ (dx, dy).

    Parameters
    ----------
    xyOffsetList : list [[x1, y1], [x2, y2], ..., [xN, yN]]
        List of the estimated [x, y] coordinates of the beam. Location
        of the origin is not important, as long as always consistent.

    commandList : list  [[u11, u12, ..., u1M], [u21, v22, ..., u2M], ..., [uN1, ..., uNM]]
        List of actuator commands.
        M actuators
        N sets of commands, one command set for each of the N xyOffsets
        This list must be the same length as xyOffsetList.

    Returns
    -------
    transRelFwd : ndarray M x 2
        transforms motion (i.e. delta position) along EXCAM pixel grid
        to motions (i.e. delta position) of the actuators.
        Satisfies requirement 1050945.

    transRelRev : ndarray 2 x M
        transforms motions (i.e. delta position) of the actuators
        to motion (i.e. delta position) along EXCAM pixel grid

    transAbsFwd : ndarray
        transforms absolute position (including translation) on the EXCAM
        pixel grid to absolute position of the actuators

    transAbsRev : ndarray
        transforms absolute position of the actuators
        to absolute position (including translation) on the EXCAM pixel grid
    """

    # input validation
    if not isinstance(xyOffsetList, list):
        raise TypeError('xyOffsetList must be a list')
    if not isinstance(commandList, list):
        raise TypeError('commandList must be a list')

    # check each xy control point is an array length 2 of reals
    for ii, xy in enumerate(xyOffsetList):
        check.oneD_array(xy, 'xyOffsetList[' + str(ii) + ']', ValueError)
        if len(xy) != 2:
            raise ValueError('xyOffsetList' + str(ii) + 'must be length 2')
        check.real_array(xy, 'xyOffsetList[' + str(ii) + ']', TypeError)

    # check each command set is an array, has same number of actuators, and real values
    for ii, cmd in enumerate(commandList):
        check.oneD_array(cmd, 'commandList[' + str(ii) + ']', ValueError)
        if len(cmd) != len(commandList[0]):
            raise ValueError('number actuators for each command set must be the same')
        check.real_array(cmd, 'commandList[' + str(ii) + ']', TypeError)

    # number actuators
    M = len(commandList[0])
    if M < 2:
        raise ValueError('must have at least 2 actuators')

    # this checks list of vectors for consistency and converts to 2d matrix
    xyOffsetList = convert_vector_list_into_two_dim_array(xyOffsetList)
    commandList = convert_vector_list_into_two_dim_array(commandList)

    # check same number of input and output control points,
    # and at least two control points
    check.twoD_array(xyOffsetList, 'xyOffsetList', TypeError)
    check.twoD_array(commandList, 'commandList', TypeError)
    if len(xyOffsetList) != len(commandList):
        raise ValueError('xyOffsetList and commandList must be same length')
    if len(xyOffsetList) < 2:
        raise ValueError('inputs must contain at least two control points')

    # check all values are real

    # if exactly two control points and two degrees of freedom,
    # call direct similarity transform
    if len(xyOffsetList) == 2 and M == 2:
        transAbsFwd, transAbsRev = calc_similarity_matrix(xyOffsetList, commandList)

    else:
        # otherwise 3 or more control points, call matrix method
        transAbsFwd, transAbsRev = calc_affine_matrix(xyOffsetList, commandList)

    # relative motion is without translation offsets
    transRelFwd = transAbsFwd.tMat[:-1, :-1]
    transRelRev = transAbsRev.tMat[:, :-1]

    transAbsFwd = transAbsFwd.tMat
    transAbsRev = transAbsRev.tMat

    return transRelFwd, transRelRev, transAbsFwd, transAbsRev

def calc_similarity_matrix(xyOffsetList, commandList):
    """
    Compute the similarity transformatin matrix for the specific case
    of two control points and four unknowns:
    scale, rotation, translation (x, y)

    see doc string for calc_trans_matrix()

    Parameters
    ----------
    xyOffsetList: list of two output control points
        [[u1, v1], [u2, v2]]
    commandList : list of two input control points
        [[x1, y1], [x2, y2]]

    Returns
    -------
    transAbsFwd : array_like
        3x3 transformation matrix used to convert beam offsets at ExCAM
        (in units of ExCAM pixels) into actuator commands (in same units as
        used in commandList). The form is as described above.
        [u, v, 1].T = T @ [x, y, 1].T

    transAbsRev : array_like
        The 3x3 transformation matrix going the opposite direction of
        transAbsFwd. That is, it converts actuator commands into beam offsets
        at ExCAM.
        [x, y] = T @ [u, v, 1].T

    """

    # inputs validated in calc_trans_matrix() above

    # deal the input and output coordinates for the control points
    u1, v1, u2, v2 = np.array(commandList).flatten()
    x1, y1, x2, y2 = np.array(xyOffsetList).flatten()

    # explicit algebraic solution, see notes
    den = (u1 - u2)**2 + (v1 - v2)**2
    if np.abs(den)/np.sum(np.array([u1, u2, v1, v2])**2) < 1e-8:
        raise ValueError('rank deficient command list')

    C = ((u1-u2)*(x1-x2) + (v1-v2)*(y1-y2))/den
    S = ((u1-u2)*(y1-y2) - (v1-v2)*(x1-x2))/den
    tx = (u1*u1*x2 + u2*u2*x1 + v1*v1*x2 + v2*v2*x1 \
          - u1*u2*x1 - u1*u2*x2 + u1*v2*y1 - u2*v1*y1 \
          - v1*v2*x1 - u1*v2*y2 + u2*v1*y2 - v1*v2*x2) \
          /den
    ty = (u1*u1*y2 + u2*u2*y1 + v1*v1*y2 + v2*v2*y1 \
          - u1*u2*y1 - u1*v2*x1 + u2*v1*x1 - u1*u2*y2 \
          + u1*v2*x2 - u2*v1*x2 - v1*v2*y1 - v1*v2*y2) \
          /den

    # T
    tformReverse = np.array([[C, -S, tx], [S, C, ty], [0, 0, 1]])
    try:
        tform = linalg.inv(tformReverse)
    except LinAlgError:
        raise ValueError('singular matrix: xyOffsetList has no movement')

    transAbsRev = CtransAbsRev(tformReverse[:-1, :])
    transAbsFwd = CtransAbsFwd(tform)

    return transAbsFwd, transAbsRev

def calc_affine_matrix(xyOffsetList, CommandList):
    """
    Compute the transformation matrix for three or more control points and two
    or more actuators.

    see doc string for calc_trans_matrix()

    Parameters
    ----------
    xyOffsetList: list of output x,y coordinates for N control points
        [[x1, y1], [x2, y2], ..., [xN, yN]]
    CommandList : list of commands for each of N control points, M actuators
        [[u11, u12, ..., u1M], [u21, u22, ..., u2M], ..., [uN1, uN2, ..., uNM]]

    Returns
    -------
    transAbsFwd : array_like
        3x3 transformation matrix used to convert beam offsets at ExCAM
        (in units of ExCAM pixels) into actuator commands (in same units as
        used in commandList). The form is as described above.
        [output u, v] = [input x, y] * T

    transAbsRev : array_like
        The 3x3 transformation matrix going the opposite direction of
        transAbsFwd. That is, it converts actuator commands into beam offsets
        at ExCAM.
        [input x, y] = [output u, v] * Tinv

    """

    # inputs validated in calc_trans_matrix() above

    # N = number of measured (x, y) control points (e.g. EXCAM coordinates)
    N = len(xyOffsetList)

    # M = number of actuators, consistency validated above
    M = len(CommandList[0])

    # calculate transAbsRev first
    # Tinv = [ax1 ay1
    #         ...
    #         axM ayM
    #         tx  ty]
    # Tinv.shape = (M+1, 2)

    # start with p.10 eqn 2, U1 @ [[Ax Ay]; [tx ty]] = xyOffsetList
    U1 = np.column_stack((np.array(CommandList), np.ones((N,))))
    # U1.shape = (Ncmds, Macts_1), i.e. N control points rows x Macts+1 cols

    # [[Ax Ay]; [tx ty]] = inv(U1) @ xyOffsetList
    U1_inv, U1_rank = linalg.pinv(U1, return_rank=True)
    transAbsRev = U1_inv @ np.array(xyOffsetList)
    transAbsRev = transAbsRev.T

    # now, p. 11 eqn 1 is a reorder of p. 10, eqn 2
    # [x y 1].T = [transAbsRev.T; [0, 0, ..., 1]] @ u1.T
    # transAbsFwd = inv([transAbsRev.T; [0, 0, ..., 1]])
    tMat1 = np.vstack((transAbsRev, np.zeros((M+1,))))
    tMat1[-1, -1] = 1.0
    transAbsFwd, tMat1_rank = linalg.pinv(tMat1, return_rank=True)

    # check ranks of each transform
    # rank of U1 (command list) must be at least Macts
    if U1_rank < M:
        raise ValueError('Command List has rank %d, must be at least %d'%(U1_rank, M))

    # rank of tMat1 (xy + translation)
    if tMat1_rank < 3:
        raise ValueError('XY Offsets have insufficient rank')

    # create an instance of class that makes application of transforom transparent
    transAbsFwd = CtransAbsFwd(transAbsFwd)
    transAbsRev = CtransAbsRev(transAbsRev)

    # check that last row of transAbsFwd should always be [0, 0, 1]


    return transAbsFwd, transAbsRev


def calc_unmasked_pupil_image_offsets(imageList, fn_pupil_fit_params,
                                      data_path=PFO_PATH):
    """
    Estimate lateral offsets for a list of 2-D pupil images.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    imageList : list
        List of nImage 2-D arrays of the clean, unocculted pupil images.
        All 2-D arrays in the list must have the same shape.
    fn_pupil_fit_params : str
        Passed directly to pupilfit_open.fit_unmasked_pupil().
        Name of the YAML file containing the tuning parameters and other
        data used for fitting the pupil.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to the directory within this repository that
        contains pupilfit_open.py.

    Returns
    -------
    xyOffsetList : list
        List of the x- and y- offsets (from the image center pixel) estimated
        by fit_unmasked_pupil(). Each list slice corresponds to the same
        slice from imageList.
    """
    if not isinstance(imageList, list):
        raise TypeError('imageList must be a list')

    for index, image in enumerate(imageList):
        check.twoD_array(image, 'image[' + str(index) + ']', TypeError)
        image = np.atleast_2d(image)  # convert to numpy array

        if index == 0:
            imageShape0 = image.shape
        else:
            if ((image.shape[0] != imageShape0[0]) or
                    (image.shape[1] != imageShape0[1])):
                raise ValueError('Arrays in imageList must have same shape.')

    # Estimate the (x, y) offsets of the pupil in each image.
    xyOffsetList = []
    for index, image in enumerate(imageList):
        image = np.atleast_2d(image)  # convert to numpy array
        xOffsetEst, yOffsetEst, _, _ = pfo.fit_unmasked_pupil(
            image, fn_pupil_fit_params, data_path=data_path)
        xyOffsetList.append([xOffsetEst, yOffsetEst])

    return xyOffsetList


def calc_unmasked_focal_image_offsets(imageList, fnImageRef, fnPSFFIT):
    """
    Estimate lateral offsets for a list of unocculted 2-D PSFs.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    imageList : list
        List of nImage 2-D arrays of the clean, unocculted PSF images. All 2-D
        arrays in the list must have the same shape.
    fnImageRef : str
        Name of FITS file containing the reference PSF to use in PSFFIT.
    fnPSFFIT : str
        Name of the YAML file containing the tuning parameter values for
        PSFFIT.

    Returns
    -------
    xyOffsetList : list
        List of the x- and y- offsets (from the image center pixel) estimated
        by PSFFIT. Each list slice corresponds to the same slice from
        imageList.
    """
    if not isinstance(imageList, list):
        raise TypeError('imageList must be a list')

    for index, image in enumerate(imageList):
        check.twoD_array(image, 'image[' + str(index) + ']', TypeError)
        image = np.atleast_2d(image)  # convert to numpy array

        if index == 0:
            imageShape0 = image.shape
        else:
            if ((image.shape[0] != imageShape0[0]) or
                    (image.shape[1] != imageShape0[1])):
                raise ValueError('Arrays in imageList must have same shape.')

    imageRef = fits.getdata(os.path.expandvars(fnImageRef))
    check.twoD_array(imageRef, 'imageRef', TypeError)
    imageRef = insertinto(imageRef, imageShape0)

    # Estimate the (x, y) offsets of the PSF in each image.
    xyOffsetList = []
    for index, image in enumerate(imageList):
        image = np.atleast_2d(image)  # convert to numpy array
        yxShiftEst, _ = psffit(imageRef, image, fnPSFFIT)
        xyOffset = [yxShiftEst[1], yxShiftEst[0]]
        xyOffsetList.append(xyOffset)

    return xyOffsetList


def calc_circular_mask_offsets(imageList, fnTuning):
    """
    Estimate lateral offsets for a circle in a list of images.

    Fits a circle to a mask that stands out against a brighter or fainter
    background. Can be used with either pupil plane or focal plane images.

    This function is a wrapper to util.fit_circle(), which is a
    wrapper for skimage.transform.hough_circle().

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    imageList : list
        List of nImage 2-D arrays of the clean mask images. All 2-D
        arrays in the list must have the same shape.
    fnTuning : str
        Name of YAML file containing the tuning parameters for the circle fit.
        The tuning parameters are:
            thresh : float, value to use when thresholding the image
            min_radius : int, min radius in pixels for hough_circle()
            max_radius : int, min radius in pixels for hough_circle()

    Returns
    -------
    xyOffsetList : list
        List of the x- and y- offsets (from the image center pixel) estimated
        by PSFFIT. Each list slice corresponds to the same slice from
        imageList.
    """
    if not isinstance(imageList, list):
        raise TypeError('imageList must be a list')
    for index, image in enumerate(imageList):
        check.twoD_array(image, 'image[' + str(index) + ']', TypeError)
        image = np.atleast_2d(image)  # convert to numpy array

        if index == 0:
            imageShape0 = image.shape
        else:
            if (image.shape[0] != imageShape0[0]) or \
               (image.shape[1] != imageShape0[1]):
                raise ValueError('Arrays in imageList must have same shape.')

    inp = loadyaml.loadyaml(fnTuning)
    thresh = inp['thresh']
    min_radius = inp['min_radius']
    max_radius = inp['max_radius']
    edge_sigma = inp['edge_sigma']

    # Estimate the (x, y) offsets of the PSF in each image.
    xyOffsetList = []
    for index, image in enumerate(imageList):
        image = np.atleast_2d(image)  # convert to numpy array
        imageThresh = np.zeros_like(image)
        imageThresh[image/np.max(image) >= thresh] = 1

        xOffsetEst, yOffsetEst, _ = fit_shapes.fit_circle(
            imageThresh, min_radius, max_radius, edge_sigma=edge_sigma)
        xyOffsetList.append([xOffsetEst, yOffsetEst])

    return xyOffsetList

def convert_vector_list_into_two_dim_array(vectorList):
    """
    Convert a list of vectors into rows in a 2-D array.

    Parameters
    ----------
    vectorList : list
        List of 1-D array_like vectors of numbers.
        The vectors must all have the same length.

    Returns
    -------
    matFromVecs : numpy ndarray
        2-D numpy array of vertically concatenated row vectors.
    """
    # Checks
    if not isinstance(vectorList, list):
        raise TypeError('vectorList must be a list')
    for index, vector in enumerate(vectorList):
        check.oneD_array(vector, 'vector[' + str(index) + ']', TypeError)
        if index == 0:
            vectorLen = len(vector)
        else:
            if len(vector) != vectorLen:
                raise ValueError('Arrays in vectorList must have same shape.')

    # Conversion from list into 2-D array
    matFromVecs = np.zeros((len(vectorList), vectorLen))
    for index, _ in enumerate(vectorList):
        matFromVecs[index, :] = vectorList[index]

    return matFromVecs


#----------
# Wrappers
#----------

def calc_trans_matrix_from_psf_images(imageList, commandList, fnImageRef,
                                      fnPSFFIT):

    """
    Compute the matrix that transforms star offsets into fold mirror commands,
    starting from a list of unocculted 2D PSFs and the commands that went
    with each.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    imageList : list
        List of nImage 2-D arrays of the clean, unocculted PSF images. All 2-D
        arrays in the list must have the same shape.
    commandList : list
        List of 1-D vectors of actuator commands. Each slice corresponds to
        the same slice of xyOffsetList. This list must be the same length as
        imageList. Each slice of commandList must have the same length.
    fnImageRef : str
        Name of FITS file containing the reference PSF to use in PSFFIT.
    fnPSFFIT : str
        Name of the YAML file containing the tuning parameter values for
        PSFFIT.

    Returns
    -------
    transRelFwd : ndarray M x 2
        transforms motion (i.e. delta position) along EXCAM pixel grid
        to motions (i.e. delta position) of the actuators
        satisfies requirement 1050945

    transRelRev : ndarray 2 x M
        transforms motions (i.e. delta position) of the actuators
        to motion (i.e. delta position) along EXCAM pixel grid

    transAbsFwd : ndarray
        transforms absolute position (including translation) on the EXCAM
        pixel grid to absolute position of the actuators

    transAbsRev : ndarray
        transforms absolute position of the actuators
        to absolute position (including translation) on the EXCAM pixel grid

    """
    # Checks on imageList
    if not isinstance(imageList, list):
        raise TypeError('imageList must be a list')
    for index, image in enumerate(imageList):
        check.twoD_array(image, 'image[' + str(index) + ']', TypeError)
        image = np.atleast_2d(image)  # convert to numpy array

        if index == 0:
            imageShape0 = image.shape
        else:
            if (image.shape[0] != imageShape0[0]) or \
               (image.shape[1] != imageShape0[1]):
                raise ValueError('Arrays in imageList must have same shape.')

    # Checks on commandList
    if not isinstance(commandList, list):
        raise TypeError('commandList must be a list')
    for index, command in enumerate(commandList):
        check.oneD_array(command, 'command[' + str(index) + ']', TypeError)
        if index == 0:
            commandLength = len(command)
        else:
            if len(command) != commandLength:
                raise ValueError('Vectors in commandList must be same shape.')
    if len(commandList) != len(imageList):
        raise ValueError('commandList and imageList must have same length')

    xyOffsetList = calc_unmasked_focal_image_offsets(imageList,
                                                     fnImageRef,
                                                     fnPSFFIT)
    return calc_trans_matrix(xyOffsetList, commandList)


def calc_trans_matrix_from_circ_mask_images(imageList, commandList, fnTuning):
    """
    Compute the matrix that transforms star offsets into actuator commands,
    starting from a list of images of circular masks and the commands that
    went with each.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    If imageList consists of pupil plane images, it is a good idea
    (to make tuning of the threshold value simpler) to run ampthresh()
    on those images before putting them into imageList(). This function
    does not call ampthresh() itself because it can be used on either
    pupil plane or focal plane images.

    Parameters
    ----------
    imageList : list
        List of nImage 2-D arrays of the clean mask images. All 2-D
        arrays in the list must have the same shape.
    commandList : list
        List of 1-D vectors of actuator commands. Each slice corresponds to
        the same slice of xyOffsetList. This list must be the same length as
        imageList. Each slice of commandList must have the same length.
    fnTuning : str
        Name of YAML file containing the tuning parameters for the circle fit.
        The tuning parameters are:
            thresh : float, value to use when thresholding the image
            min_radius : int, min radius in pixels for hough_circle()
            max_radius : int, min radius in pixels for hough_circle()

    Returns
    -------
    transRelFwd : ndarray M x 2
        transforms motion (i.e. delta position) along EXCAM pixel grid
        to motions (i.e. delta position) of the actuators
        satisfies requirement 1050945

    transRelRev : ndarray 2 x M
        transforms motions (i.e. delta position) of the actuators
        to motion (i.e. delta position) along EXCAM pixel grid

    transAbsFwd : ndarray
        transforms absolute position (including translation) on the EXCAM
        pixel grid to absolute position of the actuators

    transAbsRev : ndarray
        transforms absolute position of the actuators
        to absolute position (including translation) on the EXCAM pixel grid

    """
    # Checks on imageList
    if not isinstance(imageList, list):
        raise TypeError('imageList must be a list')
    for index, image in enumerate(imageList):
        check.twoD_array(image, 'image[' + str(index) + ']', TypeError)
        image = np.atleast_2d(image)  # convert to numpy array

        if index == 0:
            imageShape0 = image.shape
        else:
            if (image.shape[0] != imageShape0[0]) or \
               (image.shape[1] != imageShape0[1]):
                raise ValueError('Arrays in imageList must have same shape.')

    # Checks on commandList
    if not isinstance(commandList, list):
        raise TypeError('commandList must be a list')
    for index, command in enumerate(commandList):
        check.oneD_array(command, 'command[' + str(index) + ']', TypeError)
        if index == 0:
            commandLength = len(command)
        else:
            if len(command) != commandLength:
                raise ValueError('Vectors in commandList must be same shape.')
    if len(commandList) != len(imageList):
        raise ValueError('commandList and imageList must have same length')

    xyOffsetList = calc_circular_mask_offsets(imageList, fnTuning)

    return calc_trans_matrix(xyOffsetList, commandList)


def calc_trans_matrix_from_pupil_images(
        imageList, commandList, fn_pupil_fit_params, data_path=PFO_PATH):
    """
    Compute the matrix that transforms star offsets into actuator commands,
    starting from a list of 2D pupil images and the commands that
    went with each.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    imageList : list
        List of nImage 2-D arrays of the clean, unocculted pupil images.
        All 2-D arrays in the list must have the same shape.
    commandList : list
        List of 1-D vectors of actuator commands. Each slice corresponds to
        the same slice of xyOffsetList. This list must be the same length as
        imageList. Each slice of commandList must have the same length.
    fn_pupil_fit_params : str
        Passed directly to pupilfit_open.fit_unmasked_pupil().
        Name of the YAML file containing the tuning parameters and other
        data used for fitting the pupil.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to the directory within this repository that
        contains pupilfit_open.py.

    Returns
    -------
    transRelFwd : ndarray M x 2
        transforms motion (i.e. delta position) along EXCAM pixel grid
        to motions (i.e. delta position) of the actuators
        satisfies requirement 1050945

    transRelRev : ndarray 2 x M
        transforms motions (i.e. delta position) of the actuators
        to motion (i.e. delta position) along EXCAM pixel grid

    transAbsFwd : ndarray
        transforms absolute position (including translation) on the EXCAM
        pixel grid to absolute position of the actuators

    transAbsRev : ndarray
        transforms absolute position of the actuators
        to absolute position (including translation) on the EXCAM pixel grid

    """
    # Checks on imageList
    if not isinstance(imageList, list):
        raise TypeError('imageList must be a list')
    for index, image in enumerate(imageList):
        check.twoD_array(image, 'image[' + str(index) + ']', TypeError)
        image = np.atleast_2d(image)  # convert to numpy array

        if index == 0:
            imageShape0 = image.shape
        else:
            if (image.shape[0] != imageShape0[0]) or \
               (image.shape[1] != imageShape0[1]):
                raise ValueError('Arrays in imageList must have same shape.')

    # Checks on commandList
    if not isinstance(commandList, list):
        raise TypeError('commandList must be a list')
    for index, command in enumerate(commandList):
        check.oneD_array(command, 'command[' + str(index) + ']', TypeError)
        if index == 0:
            commandLength = len(command)
        else:
            if len(command) != commandLength:
                raise ValueError('Vectors in commandList must be same shape.')
    if len(commandList) != len(imageList):
        raise ValueError('commandList and imageList must have same length')

    xyOffsetList = calc_unmasked_pupil_image_offsets(
        imageList, fn_pupil_fit_params, data_path=data_path)

    return calc_trans_matrix(xyOffsetList, commandList)


def calc_trans_matrix_from_lowfs_images(imageList, commandList, fnImageRef,
                                        fnLowfsfit=LOWFS_CONFIG_DEFAULT_FN,
                                        image_bias_dark=0):
    """
    Compute the matrix that transforms LOWFS image offsets into actuator commands,
    starting from a list of 2D LOWFS images and the commands that
    went with each.

    Filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    imageList : list
        List of nImage 2-D arrays of the clean, LOWFS images.
        All 2-D arrays in the list must have the same shape.

    commandList : list
        List of 1-D vectors of actuator commands. Each slice corresponds to
        the same slice of imageList. This list must be the same length as
        imageList. Each slice of commandList must have the same length.

    fnImageRef : str
        Filename to a fits file containing a reference LOWFS image in the primary HDU,
        The reference image must be the same shape as the images in imageList.
        The reference image is a measured or modeled LOCAM image with the pupil
        at the desired location on the camera, typically the center of the camera's
        active field, with bias and dark removed.
        The offsets between the "moving" images in imageList and the reference
        are the XY offsets used to create the transform matrix. The default
        reference image was generated by cgisim model.

    fnLowfsfit : str
        Name of the YAML file containing parameters used by lowfsfit

    image_bias_dark : 2-D real array or real scalar
        a bias + dark map (dn) to be subtracted from the each image in imageList
        If a scalar, then the scalar value is subtracted as a constant from
        each image. Bias + dark are positive, real values.
        The bias + dark map must be same shape as the images in imageList.
        The default value of zero is for the case where bias and dark have
        already been subtracted.

    Returns
    -------
    transRelFwd : ndarray M x 2
        transforms motion (i.e. delta position) along EXCAM pixel grid
        to motions (i.e. delta position) of the actuators
        satisfies requirement 1050945

    transRelRev : ndarray 2 x M
        transforms motions (i.e. delta position) of the actuators
        to motion (i.e. delta position) along EXCAM pixel grid

    transAbsFwd : ndarray
        transforms absolute position (including translation) on the EXCAM
        pixel grid to absolute position of the actuators

    transAbsRev : ndarray
        transforms absolute position of the actuators
        to absolute position (including translation) on the EXCAM pixel grid

    """
    # Checks on imageList
    if not isinstance(imageList, list):
        raise TypeError('imageList must be a list')

    # check each image is 2-D real array and same shape
    for index, image in enumerate(imageList):
        check.twoD_array(image, 'image[' + str(index) + ']', TypeError)
        image = np.atleast_2d(image)  # convert to numpy array

        if index == 0:
            imageShape0 = image.shape

        if image.shape != imageShape0:
            raise ValueError('Arrays in imageList must have same shape.')

    # Checks on commandList
    if not isinstance(commandList, list):
        raise TypeError('commandList must be a list')
    for index, command in enumerate(commandList):
        check.oneD_array(command, 'command[' + str(index) + ']', TypeError)
        if index == 0:
            commandLength = len(command)
        else:
            if len(command) != commandLength:
                raise ValueError('Vectors in commandList must be same shape.')
    if len(commandList) != len(imageList):
        raise ValueError('commandList and imageList must have same length')

    # checks on fnImageRef and fnLowfsfit, are done in lowfsfit

    # check on image_bias_dark
    # this forces a scalar to 2-D array:
    if isinstance(image_bias_dark, numbers.Number):
        image_bias_dark = image_bias_dark * np.ones(imageShape0)

    # check image_bias_dark is 2-D, real, and same shape as images
    check.twoD_array(image_bias_dark, 'image_bias_dark', TypeError)
    image_bias_dark = np.atleast_2d(image_bias_dark)  # convert to numpy array
    if image_bias_dark.shape != imageShape0:
        raise ValueError('image_bias_dark must have same shape as input images.')

    # subtract bias_dark from each moving image
    for img in imageList:
        img -= image_bias_dark

    # Estimate the (x, y) offsets of the pupil in each image.
    xyOffsetList = lowfsfit.locate_img(imageList, fnImageRef, fnLowfsfit)

    return calc_trans_matrix(xyOffsetList, commandList)
