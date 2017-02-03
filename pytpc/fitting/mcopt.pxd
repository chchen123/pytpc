from libcpp.vector cimport vector as cppvec
from libcpp.map cimport map as cppmap
cimport armadillo as arma

cdef extern from "mcopt/mcopt.h" namespace "mcopt":
    ctypedef unsigned short pad_t


    cdef cppclass Gas:
        Gas(const cppvec[double]& eloss, const cppvec[double]& enVsZ) except+
        double energyLoss(const double energy) except+
        double energyFromVertexZPosition(const double z) except+


    cdef cppclass Track:
        Track() except+
        arma.mat getMatrix() except+
        arma.mat getPositionMatrix() except+
        arma.vec getEnergyVector() except+
        size_t numPts() except+


    cdef cppclass Tracker:
        Tracker(unsigned massNum, unsigned chargeNum, const Gas* gas,
                arma.vec& efield, arma.vec& bfield) except+
        Track trackParticle(const double x0, const double y0, const double z0,
                            const double enu0,  const double azi0, const double pol0) except+
        unsigned int getMassNum()
        unsigned int getChargeNum()
        arma.vec getEfield()
        void setEfield(const arma.vec& ef)
        arma.vec getBfield()
        void setBfield(const arma.vec& bf)


    cdef cppclass PadPlane:
        PadPlane(const arma.Mat[pad_t]& lt, const double xLB, const double xDelta,
                 const double yLB, const double yDelta, const double rotAngle) except+
        pad_t getPadNumberFromCoordinates(const double x, const double y) except+
        @staticmethod
        cppvec[cppvec[cppvec[double]]] generatePadCoordinates(const double rotation_angle)


    cdef cppclass EventGenerator:
        EventGenerator(const PadPlane* pads, const arma.vec& vd, const double clock, const double shape,
                       const unsigned massNum, const double ioniz, const double micromegasGain,
                       const double electronicsGain, const double tilt, const double diffSigma) except+

        cppmap[pad_t, arma.vec] makeEvent(const arma.mat& pos, const arma.vec& en) except+
        arma.mat makePeaksTableFromSimulation(const arma.mat& pos, const arma.vec& en) except+
        arma.vec makeMeshSignal(const arma.mat& pos, const arma.vec& en) except+
        arma.vec makeHitPattern(const arma.mat& pos, const arma.vec& en) except+

        arma.vec vd
        unsigned massNum
        double ioniz
        double micromegasGain
        double electronicsGain
        double tilt
        double diffSigma
        double clock
        double shape


    cdef cppclass MCminimizeResult:
        MCminimizeResult() except +
        arma.vec ctr
        arma.mat allParams
        arma.mat minChis
        arma.vec goodParamIdx


    cdef cppclass Chi2Set:
        Chi2Set() except+
        double posChi2
        double enChi2


    cdef cppclass BeamLocationEstimator:
        BeamLocationEstimator(const double xslope, const double xint, const double yslope, const double yint) except+
        double findX(const double z)
        arma.vec findX(const arma.vec z)

        double findY(const double z)
        arma.vec findY(const arma.vec z)


    cdef cppclass MCminimizer:
        MCminimizer(const Tracker* tracker, const EventGenerator* evtgen,
                    const unsigned numIters, const unsigned numPts, const double redFactor) except+

        arma.mat makeParams(const arma.vec& ctr, const arma.vec& sigma, const unsigned numSets,
                            const arma.vec& mins, const arma.vec& maxes) except+
        arma.mat findPositionDeviations(const arma.mat& simPos, const arma.mat& expPos) except+
        arma.vec findHitPatternDeviation(const arma.mat& simPos, const arma.vec& simEn, const arma.vec& expHits) except+
        MCminimizeResult minimize(const arma.vec& ctr0, const arma.vec& sigma0, const arma.mat& expPos,
                                  const arma.vec& expMesh, const BeamLocationEstimator& beamloc) except+
        Chi2Set runTrack(const arma.vec& params, const arma.mat& expPos, const arma.vec& expHits) except+
        arma.mat runTracks(const arma.mat& params, const arma.mat& expPos, const arma.vec& expHits) except+

        bint posChi2Enabled
        bint enChi2Enabled
        bint vertChi2Enabled

        double posChi2Norm
        double enChi2NormFraction
        double vertChi2Norm

        unsigned numIters
        unsigned numPts
        double redFactor
