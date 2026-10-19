# Purpose of MOTIONFIT Module

MOTIONFIT calculates the transformation matrix that converts beam offsets at ExCAM into actuator commands (e.g., for FSM or OTA TTF). The images of the beam can be of either unocculted PSFs or the unmasked pupil; separate functions exist to measure offsets for them using functions from the PSFFIT and PUPILFIT modules, respectively. The piston term for the computed actuator commands is removed automatically because of the Moore Penrose pseudoinverse used to compute the transformation matrix.


# Usage

The matrices `transMatrix` and `transMatrixReverse` returned by `motionfit.calc_trans_matrix()` are related to the actuator commands and the x- and y-offsets as follows:

- `cmdVec = transMatrix @ xyOffsetVec`
- `xyOffsetVec = transMatrixReverse @ cmdVec`,

where `@` is the matrix multiplication operator, and `cmdVec` and `xyOffsetVec` are column vectors of the actuator commands and lateral offsets, respectively. The lateral offset vector contains the x-value first and the y-value second.


# Related Requirement(s)

_1050945: L5 CTC GSW - Computation of FSM-EXCAM transformation_

Given 1) a set of clean unocculted focal-plane images and 2) a set of FSM offsets, each matched to a clean image, the CTC GSW shall compute a transformation matrix from motions along the EXCAM pixel grid to motions of the FSM actuators which removes FSM piston.
