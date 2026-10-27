"""Frozen scalar settings reported in the manuscript.

Every value below was selected on the FSC-147 *validation* split and then frozen;
nothing here is re-selected on FSC-147 test or on CARPK.
"""

# Prototype--contrastive kernel masking, Eq. (4)--(6) of the manuscript.
#   gamma : side ratio of the central region (0 < gamma < 1)
#   alpha : floor of the content mask
#   eta   : exponent applied to the content mask
PCM = dict(gamma=0.6, alpha=0.05, eta=2)

# Complementary peak--mass readout, Eq. (7)--(10).
#   beta    : peak neighbourhood radius as a fraction of the exemplar box radius
#   t_peak  : relative peak threshold (fraction of the maximum of the map)
#   guard   : high-count guard C_g; above it the mass estimate is a lower bound
PMR = dict(beta=0.25, t_peak=0.35, guard=300.0)

# Density-adaptive tiling thresholds, applied to the 1x integral probe count.
#   DAT (integral readout)      theta = (19, 31)
#   full PCM + DAT + PMR stack  theta = (15, 31)
DAT_INTEGRAL = dict(theta1=19, theta2=31)
DAT = dict(theta1=15, theta2=31)
