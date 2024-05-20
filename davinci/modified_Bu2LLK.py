###############################################################################
# (c) Copyright 2000-2019 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
__author__ = "Patrick Koppenburg, Alex Shires, Thomas Blake, Luca Pescatore, Simone Bifani, Yasmine Amhis, Paula Alvarez Cartelle, Harry Cliff, Rafael Silva Coutinho, Guido Andreassi, Mick Mulder"
__date__ = "16/06/2014"
__version__ = "$Revision: 3 $"

__all__ = ("Bu2LLKConf", "default_config")

"""
  B --> ll K selections for:
  B --s> ee K  versus  B --> mumu K
  B --s> ee pi  versus  B --> mumu pi
  B --> ee K*  versus  B --> mumu K*
  B --> ee K1  versus  B --> mumu K1
  B --> ee phi  versus  B --> mumu phi
  Lb --> ee Lambda(*)  versus  Lb --> mumu Lambda(*)
  B --> gamma K* , B --> gamma phi  and  Lb --> gamma Lambda(*)  with converted photons
  B --> emu K(+/0), B--> emu K*(0/+) and B--> emu Phi, Lb--> Lambda(*) emu, Lb--> Lambda(*) taumu with OS and SS leptons
"""

daughter_locations = {
    # OPPOSITE SIGN
    # 3-body
    "[Beauty -> ^X+  (X0 ->  l+  l-)]CC": "{0}H",
    "[Beauty ->  X+  (X0 -> ^l+  l-)]CC": "{0}L1",
    "[Beauty ->  X+  (X0 ->  l+ ^l-)]CC": "{0}L2",
    "[Beauty ->  X+ ^(X0 ->  l+  l-)]CC": "{0}LL",
    # 5-body (quasi 3-body)
    "[Beauty -> (X+ -> ^X+  X+  X-) (X0 ->  l+  l-)]CC": "{0}H1",
    "[Beauty -> (X+ ->  X+ ^X+  X-) (X0 ->  l+  l-)]CC": "{0}H2",
    "[Beauty -> (X+ ->  X+  X+ ^X-) (X0 ->  l+  l-)]CC": "{0}H3",
    # 5-body
    "[Beauty -> (X+ -> ^X+  (X0 ->  X+  X-)) (X0 ->  l+  l-)]CC": "{0}H1",
    "[Beauty -> (X+ ->  X+  (X0 -> ^X+  X-)) (X0 ->  l+  l-)]CC": "{0}H2",
    "[Beauty -> (X+ ->  X+  (X0 ->  X+ ^X-)) (X0 ->  l+  l-)]CC": "{0}H3",
    "[Beauty -> (X+ ->  X+  (X0 ->  X+  X-)) (X0 -> ^l+  l-)]CC": "{0}L1",
    "[Beauty -> (X+ ->  X+  (X0 ->  X+  X-)) (X0 ->  l+ ^l-)]CC": "{0}L2",
    "[Beauty -> (X+ ->  X+ ^(X0 ->  X+  X-)) (X0 ->  l+  l-)]CC": "{0}HH",
    # 4-body with a strange particle in the final state
    "[Beauty ->  (X0 -> ^Xs  X-)  (X0 ->  l+  l-)]CC": "{0}H1",
    "[Beauty ->  (X0 ->  Xs ^X-)  (X0 ->  l+  l-)]CC": "{0}H2",
    "[Beauty ->  (X0 ->  Xs  X-)  (X0 -> ^l+  l-)]CC": "{0}L1",
    "[Beauty ->  (X0 ->  Xs  X-)  (X0 ->  l+ ^l-)]CC": "{0}L2",
    "[Beauty -> ^(X0 ->  Xs  X-)  (X0 ->  l+  l-)]CC": "{0}HH",
    "[Beauty ->  (X0 ->  Xs  X-) ^(X0 ->  l+  l-)]CC": "{0}LL",
    # 4-body with two pions in the final state
    "[Beauty ->  (X0 -> ^pi+  pi-)  (X0 ->  l+  l-)]CC": "{0}H1",
    "[Beauty ->  (X0 ->  pi+ ^pi-)  (X0 ->  l+  l-)]CC": "{0}H2",
    "[Beauty ->  (X0 ->  pi+  pi-)  (X0 -> ^l+  l-)]CC": "{0}L1",
    "[Beauty ->  (X0 ->  pi+  pi-)  (X0 ->  l+ ^l-)]CC": "{0}L2",
    "[Beauty -> ^(X0 ->  pi+  pi-)  (X0 ->  l+  l-)]CC": "{0}HH",
    "[Beauty ->  (X0 ->  pi+  pi-) ^(X0 ->  l+  l-)]CC": "{0}LL",
    # 4-body with p and pi in the final state. Here the names are kept generic (H1,H2,L1,L2), for uniformity with strippingof other years. This is needed for Lb->Lll analyses.
    "[Beauty ->  (X0 -> ^p+  pi-)  (X0 ->  l+  l-)]CC": "{0}H1",
    "[Beauty ->  (X0 ->  p+ ^pi-)  (X0 ->  l+  l-)]CC": "{0}H2",
    "[Beauty ->  (X0 ->  p+  pi-)  (X0 -> ^l+  l-)]CC": "{0}L1",
    "[Beauty ->  (X0 ->  p+  pi-)  (X0 ->  l+ ^l-)]CC": "{0}L2",
    "[Beauty -> ^(X0 ->  p+  pi-)  (X0 ->  l+  l-)]CC": "{0}HH",
    "[Beauty ->  (X0 ->  p+  pi-) ^(X0 ->  l+  l-)]CC": "{0}LL",
    # 7-body (quasi 4-body)
    "[Beauty ->  (X0 ->  X+  X-)  (X0 ->  l+  (l- -> ^X-  X-  X+))]CC": "{0}L21",
    "[Beauty ->  (X0 ->  X+  X-)  (X0 ->  l+  (l- ->  X- ^X-  X+))]CC": "{0}L22",
    "[Beauty ->  (X0 ->  X+  X-)  (X0 ->  l+  (l- ->  X-  X- ^X+))]CC": "{0}L23",
    # SAME SIGN
    # 3-body
    "[Beauty -> ^X+  (X+ ->  l+  l+)]CC": "{0}H",
    "[Beauty ->  X+  (X+ -> ^l+  l+)]CC": "{0}L1",
    "[Beauty ->  X+  (X+ ->  l+ ^l+)]CC": "{0}L2",
    "[Beauty ->  X+ ^(X+ ->  l+  l+)]CC": "{0}LL",
    # 5-body (quasi 3-body)
    "[Beauty -> (X+ -> ^X+  X+  X-) (X+ ->  l+  l+)]CC": "{0}H1",
    "[Beauty -> (X+ ->  X+ ^X+  X-) (X+ ->  l+  l+)]CC": "{0}H2",
    "[Beauty -> (X+ ->  X+  X+ ^X-) (X+ ->  l+  l+)]CC": "{0}H3",
    # 5-body
    "[Beauty -> (X+ -> ^X+  (X0 ->  X+  X-)) (X+ ->  l+  l+)]CC": "{0}H1",
    "[Beauty -> (X+ ->  X+  (X0 -> ^X+  X-)) (X+ ->  l+  l+)]CC": "{0}H2",
    "[Beauty -> (X+ ->  X+  (X0 ->  X+ ^X-)) (X+ ->  l+  l+)]CC": "{0}H3",
    "[Beauty -> (X+ ->  X+  (X0 ->  X+  X-)) (X+ -> ^l+  l+)]CC": "{0}L1",
    "[Beauty -> (X+ ->  X+  (X0 ->  X+  X-)) (X+ ->  l+ ^l+)]CC": "{0}L2",
    "[Beauty -> (X+ ->  X+ ^(X0 ->  X+  X-)) (X+ ->  l+  l+)]CC": "{0}HH",
    # 4-body with a strange particle in the final state
    "[Beauty ->  (X0 -> ^Xs  X-)  (X ->  l+  l+)]CC": "{0}H1",
    "[Beauty ->  (X0 -> ^Xs  X-)  (X ->  l-  l-)]CC": "{0}H1",
    "[Beauty ->  (X0 ->  Xs ^X-)  (X ->  l+  l+)]CC": "{0}H2",
    "[Beauty ->  (X0 ->  Xs ^X-)  (X ->  l-  l-)]CC": "{0}H2",
    "[Beauty ->  (X0 ->  Xs  X-)  (X -> ^l+  l+)]CC": "{0}L1",
    "[Beauty ->  (X0 ->  Xs  X-)  (X -> ^l-  l-)]CC": "{0}L1",
    "[Beauty ->  (X0 ->  Xs  X-)  (X ->  l+ ^l+)]CC": "{0}L2",
    "[Beauty ->  (X0 ->  Xs  X-)  (X ->  l- ^l-)]CC": "{0}L2",
    "[Beauty -> ^(X0 ->  Xs  X-)  (X ->  l+  l+)]CC": "{0}HH",
    "[Beauty -> ^(X0 ->  Xs  X-)  (X ->  l-  l-)]CC": "{0}HH",
    "[Beauty ->  (X0 ->  Xs  X-) ^(X ->  l+  l+)]CC": "{0}LL",
    "[Beauty ->  (X0 ->  Xs  X-) ^(X ->  l-  l-)]CC": "{0}LL",
    # 4-body with two pions in the final state
    "[Beauty ->  (X0 -> ^pi+  pi-)  (X+ ->  l+  l+)]CC": "{0}H1",
    "[Beauty ->  (X0 ->  pi+ ^pi-)  (X+ ->  l+  l+)]CC": "{0}H2",
    "[Beauty ->  (X0 ->  pi+  pi-)  (X+ -> ^l+  l+)]CC": "{0}L1",
    "[Beauty ->  (X0 ->  pi+  pi-)  (X+ ->  l+ ^l+)]CC": "{0}L2",
    "[Beauty -> ^(X0 ->  pi+  pi-)  (X+ ->  l+  l+)]CC": "{0}HH",
    "[Beauty ->  (X0 ->  pi+  pi-) ^(X+ ->  l+  l+)]CC": "{0}LL",
    # 4-body with p and pi in the final state. Here the names are kept generic (H1,H2,L1,L2), for uniformity with strippingof other years. This is needed for Lb->Lll analyses.
    "[Beauty ->  (X0 -> ^p+  pi-)  (X ->  l+  l+)]CC": "{0}H1",
    "[Beauty ->  (X0 -> ^p+  pi-)  (X ->  l-  l-)]CC": "{0}H1",
    "[Beauty ->  (X0 ->  p+ ^pi-)  (X ->  l+  l+)]CC": "{0}H2",
    "[Beauty ->  (X0 ->  p+ ^pi-)  (X ->  l-  l-)]CC": "{0}H2",
    "[Beauty ->  (X0 ->  p+  pi-)  (X -> ^l+  l+)]CC": "{0}L1",
    "[Beauty ->  (X0 ->  p+  pi-)  (X -> ^l-  l-)]CC": "{0}L1",
    "[Beauty ->  (X0 ->  p+  pi-)  (X ->  l+ ^l+)]CC": "{0}L2",
    "[Beauty ->  (X0 ->  p+  pi-)  (X ->  l- ^l-)]CC": "{0}L2",
    "[Beauty -> ^(X0 ->  p+  pi-)  (X ->  l+  l+)]CC": "{0}HH",
    "[Beauty -> ^(X0 ->  p+  pi-)  (X ->  l-  l-)]CC": "{0}HH",
    "[Beauty ->  (X0 ->  p+  pi-) ^(X ->  l+  l+)]CC": "{0}LL",
    "[Beauty ->  (X0 ->  p+  pi-) ^(X ->  l-  l-)]CC": "{0}LL",
    # 7-body (quasi 4-body)
    "[Beauty ->  (X0 ->  X+  X-)  (X+ ->  l+  (l+ -> ^X+  X-  X+))]CC": "{0}L21",
    "[Beauty ->  (X0 ->  X+  X-)  (X+ ->  l+  (l+ ->  X+ ^X-  X+))]CC": "{0}L22",
    "[Beauty ->  (X0 ->  X+  X-)  (X+ ->  l+  (l+ ->  X+  X- ^X+))]CC": "{0}L23",
}

daughter_vtx_locations = {
    # OPPOSITE SIGN
    # 3-body
    "[Beauty ->  X+ ^(X0 ->  l+  l-)]CC": "{0}LL",
    # 5-body (quasi 3-body)
    "[Beauty -> ^(X+ -> X+  X+  X-) (X0 ->  l+  l-)]CC": "{0}H",
    # 5-body
    "[Beauty -> ^(X+ -> X+  (X0 ->  X+  X-)) (X0 ->  l+  l-)]CC": "{0}H",
    "[Beauty -> (X+ ->  X+ ^(X0 ->  X+  X-)) (X0 ->  l+  l-)]CC": "{0}HH",
    # 4-body
    "[Beauty -> ^(X0 ->  X+  X-)  (X0 ->  l+  l-)]CC": "{0}HH",
    "[Beauty ->  (X0 ->  X+  X-) ^(X0 ->  l+  l-)]CC": "{0}LL",
    # 7-body (quasi 4-body)
    "[Beauty ->  X  (X0 ->  l+  ^(l- -> X-  X-  X+))]CC": "{0}L",
    # SAME SIGN
    # 3-body
    "[Beauty ->  X+ ^(X+ ->  l+  l+)]CC": "{0}LL",
    # 5-body (quasi 3-body)
    "[Beauty -> ^(X+ -> X+  X+  X-) (X+ ->  l+  l+)]CC": "{0}H",
    # 5-body
    "[Beauty -> ^(X+ -> X+  (X0 ->  X+  X-)) (X+ ->  l+  l+)]CC": "{0}H",
    "[Beauty -> (X+ ->  X+ ^(X0 ->  X+  X-)) (X+ ->  l+  l+)]CC": "{0}HH",
    # 4-body
    "[Beauty -> ^(X0 ->  X+  X-)  (X ->  l+  l+)]CC": "{0}HH",
    "[Beauty ->  (X0 ->  X+  X-) ^(X ->  l+  l+)]CC": "{0}LL",
    # 7-body (quasi 4-body)
    "[Beauty ->  X  (X+ ->  l+  ^(l+ -> X-  X-  X+))]CC": "{0}L",
}


default_config = {
    "NAME": "Bu2LLK",
    "BUILDERTYPE": "Bu2LLKConf",
    "CONFIG": {
        "BFlightCHI2": 100,
        "BDIRA": 0.9995,
        "BIPCHI2": 25,
        "BVertexCHI2": 9,
        "DiLeptonPT": 0,
        "DiLeptonFDCHI2": 16,
        "DiLeptonIPCHI2": 0,
        "LeptonIPCHI2": 9,
        "LeptonPT": 300,
        "TauPT": 0,
        "TauVCHI2DOF": 150,
        "KaonIPCHI2": 9,
        "KaonPT": 400,
        "KstarPVertexCHI2": 25,
        "KstarPMassWindow": 300,
        "KstarPADOCACHI2": 30,
        "DiHadronMass": 2600,
        "UpperMass": 5500,
        "BMassWindow": 1500,
        "BMassWindowTau": 5000,
        "PIDe": 0,
        "Trk_Chi2": 3,
        "Trk_GhostProb": 0.4,
        "K1_MassWindow_Lo": 0,
        "K1_MassWindow_Hi": 6000,
        "K1_VtxChi2": 12,
        "K1_SumPTHad": 800,
        "K1_SumIPChi2Had": 48.0,
        "Bu2eeLinePrescale": 1,
        "Bu2eeLine2Prescale": 1,
        "Bu2eeLine3Prescale": 1,
        "Bu2mmLinePrescale": 1,
        "Bu2meLinePrescale": 1,
        "Bu2meSSLinePrescale": 1,
        "Bu2mtLinePrescale": 1,
        "Bu2mtSSLinePrescale": 1,
        "RelatedInfoTools": [
            {
                "Type": "RelInfoVertexIsolation",
                "Location": "VertexIsoInfo",
                "IgnoreUnmatchedDescriptors": True,
                "DaughterLocations": {
                    key: val.format("VertexIsoInfo")
                    for key, val in daughter_vtx_locations.items()
                },
            },
            {
                "Type": "RelInfoVertexIsolationBDT",
                "Location": "VertexIsoBDTInfo",
                "IgnoreUnmatchedDescriptors": True,
                "DaughterLocations": {
                    key: val.format("VertexIsoBDTInfo")
                    for key, val in daughter_vtx_locations.items()
                },
            },
            {
                "Type": "RelInfoConeVariables",
                "ConeAngle": 0.5,
                "IgnoreUnmatchedDescriptors": True,
                "Location": "TrackIsoInfo05",
                "DaughterLocations": {
                    key: val.format("TrackIsoInfo")
                    for key, val in daughter_locations.items()
                },
            },
            {
                "Type": "RelInfoConeIsolation",
                "ConeSize": 0.5,
                "IgnoreUnmatchedDescriptors": True,
                "Location": "ConeIsoInfo05",
                "DaughterLocations": {
                    key: val.format("ConeIsoInfo")
                    for key, val in daughter_locations.items()
                },
            },
            {
                "Type": "RelInfoTrackIsolationBDT",
                "IgnoreUnmatchedDescriptors": True,
                # Use the BDT with 9 input variables
                # This requires that the "Variables" value is set to 2
                "Variables": 2,
                "WeightsFile": "BsMuMu_TrackIsolationBDT9vars_v1r4.xml",
                "Location": "TrackIsoBDTInfo",
                "DaughterLocations": {
                    key: val.format("TrackIsoBDTInfo")
                    for key, val in daughter_locations.items()
                },
            },
            {
                "Type": "RelInfoBs2MuMuTrackIsolations",
                "IgnoreUnmatchedDescriptors": True,
                "Location": "TrackIsoBs2MMInfo",
                "DaughterLocations": {
                    key: val.format("TrackIsoBs2MMInfo")
                    for key, val in daughter_locations.items()
                },
            },
            {
                "Type": "RelInfoConeVariables",
                "ConeAngle": 1.0,
                "Location": "TrackIsoInfo10",
            },
            {
                "Type": "RelInfoConeVariables",
                "ConeAngle": 1.5,
                "Location": "TrackIsoInfo15",
            },
            {
                "Type": "RelInfoConeVariables",
                "ConeAngle": 2.0,
                "Location": "TrackIsoInfo20",
            },
            {
                "Type": "RelInfoConeIsolation",
                "ConeSize": 1.0,
                "Location": "ConeIsoInfo10",
            },
            {
                "Type": "RelInfoConeIsolation",
                "ConeSize": 1.5,
                "Location": "ConeIsoInfo15",
            },
            {
                "Type": "RelInfoConeIsolation",
                "ConeSize": 2.0,
                "Location": "ConeIsoInfo20",
            },
        ],
    },
    "WGs": ["RD"],
    "STREAMS": ["Leptonic"],
}


from Gaudi.Configuration import *
from GaudiConfUtils.ConfigurableGenerators import (
    FilterDesktop,
    CombineParticles,
    DaVinci__N3BodyDecays,
)
from PhysSelPython.Wrappers import (
    Selection,
    DataOnDemand,
    MergedSelection,
    AutomaticData,
)
from StrippingConf.StrippingLine import StrippingLine
from StrippingUtils.Utils import LineBuilder


class Bu2LLKConf(LineBuilder):
    """
    Builder for R_X measurements
    """

    # now just define keys. Default values are fixed later
    __configuration_keys__ = (
        "BFlightCHI2",
        "BDIRA",
        "BIPCHI2",
        "BVertexCHI2",
        "DiLeptonPT",
        "DiLeptonFDCHI2",
        "DiLeptonIPCHI2",
        "LeptonIPCHI2",
        "LeptonPT",
        "TauPT",
        "TauVCHI2DOF",
        "KaonIPCHI2",
        "KaonPT",
        "KstarPVertexCHI2",
        "KstarPMassWindow",
        "KstarPADOCACHI2",
        "DiHadronMass",
        "UpperMass",
        "BMassWindow",
        "BMassWindowTau",
        "PIDe",
        "Trk_Chi2",
        "Trk_GhostProb",
        "K1_MassWindow_Lo",
        "K1_MassWindow_Hi",
        "K1_VtxChi2",
        "K1_SumPTHad",
        "K1_SumIPChi2Had",
        "Bu2eeLinePrescale",
        "Bu2eeLine2Prescale",
        "Bu2eeLine3Prescale",
        "Bu2mmLinePrescale",
        "Bu2meLinePrescale",
        "Bu2meSSLinePrescale",
        "Bu2mtLinePrescale",
        "Bu2mtSSLinePrescale",
        "RelatedInfoTools",
    )

    def __init__(self, name, config):
        LineBuilder.__init__(self, name, config)

        self._name = name

        mmXLine_name = name + "_mm"
        eeXLine_name = name + "_ee"
        meXLine_name = name + "_me"
        meXSSLine_name = name + "_meSS"
        mtXLine_name = name + "_mt"
        mtXSSLine_name = name + "_mtSS"

        from StandardParticles import StdLoosePions as Pions
        from StandardParticles import StdLooseKaons as Kaons
        from StandardParticles import StdLooseKstar2Kpi as Kstars
        from StandardParticles import StdLoosePhi2KK as Phis
        from StandardParticles import StdLooseKsLL as KshortsLL
        from StandardParticles import StdLooseKsDD as KshortsDD
        from StandardParticles import StdLooseLambdaLL as LambdasLL
        from StandardParticles import StdLooseLambdaDD as LambdasDD
        from StandardParticles import StdLooseLambdastar2pK as Lambdastars

        # 1 : Make K, Ks, K*, K1, Phi and Lambdas

        SelKaons = self._filterHadron(
            name="KaonsFor" + self._name, sel=Kaons, params=config
        )

        SelPions = self._filterHadron(
            name="PionsFor" + self._name, sel=Pions, params=config
        )

        SelKshortsLL = self._filterHadron(
            name="KshortsLLFor" + self._name, sel=KshortsLL, params=config
        )

        SelKshortsDD = self._filterHadron(
            name="KshortsDDFor" + self._name, sel=KshortsDD, params=config
        )

        SelKstars = self._filterHadron(
            name="KstarsFor" + self._name, sel=Kstars, params=config
        )

        SelKstarsPlusLL = self._makeKstarPlus(
            name="KstarsPlusLLFor" + self._name,
            kshorts=KshortsLL,
            pions=Pions,
            params=config,
        )

        SelKstarsPlusDD = self._makeKstarPlus(
            name="KstarsPlusDDFor" + self._name,
            kshorts=KshortsDD,
            pions=Pions,
            params=config,
        )

        SelK1s = self._makeK1(
            name="K1For" + self._name, kaons=Kaons, pions=Pions, params=config
        )

        SelPhis = self._filterHadron(
            name="PhisFor" + self._name, sel=Phis, params=config
        )

        SelLambdasLL = self._filterHadron(
            name="LambdasLLFor" + self._name, sel=LambdasLL, params=config
        )

        SelLambdasDD = self._filterHadron(
            name="LambdasDDFor" + self._name, sel=LambdasDD, params=config
        )

        SelLambdastars = self._filterHadron(
            name="LambdastarsFor" + self._name, sel=Lambdastars, params=config
        )

        # 2 : Make Dileptons

        from StandardParticles import StdDiElectronFromTracks as DiElectronsFromTracks
        from StandardParticles import StdLooseDiElectron as DiElectrons
        from StandardParticles import StdLooseDiMuon as DiMuons

        ElecID = "(PIDe > %(PIDe)s)" % config
        MuonID = "(HASMUON)&(ISMUON)"
        TauID = "(PT > %(TauPT)s)" % config

        MuE = self._makeMuE(
            "MuEFor" + self._name, params=config, electronid=ElecID, muonid=MuonID
        )
        MuE_SS = self._makeMuE(
            "MuESSFor" + self._name,
            params=config,
            electronid=ElecID,
            muonid=MuonID,
            samesign=True,
        )

        MuTau = self._makeMuTau(
            "MuTauFor" + self._name, params=config, tauid=TauID, muonid=MuonID
        )
        MuTau_SS = self._makeMuTau(
            "MuTauSSFor" + self._name,
            params=config,
            tauid=TauID,
            muonid=MuonID,
            samesign=True,
        )

        DiElectronID = "(2 == NINTREE((ABSID==11)&(PIDe > %(PIDe)s)))" % config
        DiMuonID = "(2 == NINTREE((ABSID==13)&(HASMUON)&(ISMUON)))"

        SelDiElectron = self._filterDiLepton(
            "SelDiElectronFor" + self._name,
            dilepton=DiElectrons,
            params=config,
            idcut=DiElectronID,
        )

        SelDiElectronFromTracks = self._filterDiLepton(
            "SelDiElectronFromTracksFor" + self._name,
            dilepton=DiElectronsFromTracks,
            params=config,
            idcut=DiElectronID,
        )

        SelDiMuon = self._filterDiLepton(
            "SelDiMuonsFor" + self._name,
            dilepton=DiMuons,
            params=config,
            idcut=DiMuonID,
        )

        SelMuE = self._filterDiLepton(
            "SelMuEFor" + self._name, dilepton=MuE, params=config, idcut=None
        )

        SelMuE_SS = self._filterDiLepton(
            "SelMuESSFor" + self._name, dilepton=MuE_SS, params=config, idcut=None
        )

        SelMuTau = self._filterMuTau(
            "SelMuTauFor" + self._name, dilepton=MuTau, params=config, idcut=None
        )

        SelMuTau_SS = self._filterMuTau(
            "SelMuTauSSFor" + self._name, dilepton=MuTau_SS, params=config, idcut=None
        )

        # 3 : Make Photons

        from StandardParticles import StdAllLooseGammaLL as PhotonConversion

        SelPhoton = self._filterPhotons(
            "SelPhotonFor" + self._name, photons=PhotonConversion
        )

        # 4 : Combine Particles

        SelB2eeX = self._makeB2LLX(
            eeXLine_name,
            dilepton=SelDiElectron,
            hadrons=[
                SelPions,
                SelKaons,
                SelKstars,
                SelPhis,
                SelKshortsLL,
                SelKshortsDD,
                SelLambdasLL,
                SelLambdasDD,
                SelLambdastars,
                SelKstarsPlusLL,
                SelKstarsPlusDD,
                SelK1s,
            ],
            params=config,
            masscut="ADAMASS('B+') <  %(BMassWindow)s *MeV" % config,
        )

        SelB2eeXFromTracks = self._makeB2LLX(
            eeXLine_name + "2",
            dilepton=SelDiElectronFromTracks,
            hadrons=[
                SelPions,
                SelKaons,
                SelKstars,
                SelPhis,
                SelKshortsLL,
                SelKshortsDD,
                SelLambdasLL,
                SelLambdasDD,
                SelLambdastars,
                SelKstarsPlusLL,
                SelKstarsPlusDD,
                SelK1s,
            ],
            params=config,
            masscut="ADAMASS('B+') <  %(BMassWindow)s *MeV" % config,
        )

        SelB2mmX = self._makeB2LLX(
            mmXLine_name,
            dilepton=SelDiMuon,
            hadrons=[
                SelPions,
                SelKaons,
                SelKstars,
                SelPhis,
                SelKshortsLL,
                SelKshortsDD,
                SelLambdasLL,
                SelLambdasDD,
                SelLambdastars,
                SelKstarsPlusLL,
                SelKstarsPlusDD,
                SelK1s,
            ],
            params=config,
            masscut="ADAMASS('B+') <  %(BMassWindow)s *MeV" % config,
        )

        SelB2meX = self._makeB2LLX(
            meXLine_name,
            dilepton=SelMuE,
            hadrons=[
                SelPions,
                SelKaons,
                SelKstars,
                SelPhis,
                SelKshortsLL,
                SelKshortsDD,
                SelLambdasLL,
                SelLambdasDD,
                SelLambdastars,
                SelKstarsPlusLL,
                SelKstarsPlusDD,
            ],
            params=config,
            masscut="ADAMASS('B+') <  %(BMassWindow)s *MeV" % config,
        )

        SelB2meX_SS = self._makeB2LLX(
            meXSSLine_name,
            dilepton=SelMuE_SS,
            hadrons=[
                SelPions,
                SelKaons,
                SelKstars,
                SelPhis,
                SelKshortsLL,
                SelKshortsDD,
                SelLambdasLL,
                SelLambdasDD,
                SelLambdastars,
                SelKstarsPlusLL,
                SelKstarsPlusDD,
            ],
            params=config,
            masscut="ADAMASS('B+') <  %(BMassWindow)s *MeV" % config,
        )

        SelB2mtX = self._makeB2LLX(
            mtXLine_name,
            dilepton=SelMuTau,
            hadrons=[SelLambdastars],
            params=config,
            masscut="ADAMASS('B+') <  %(BMassWindowTau)s *MeV" % config,
        )

        SelB2mtX_SS = self._makeB2LLX(
            mtXSSLine_name,
            dilepton=SelMuTau_SS,
            hadrons=[SelLambdastars],
            params=config,
            masscut="ADAMASS('B+') <  %(BMassWindowTau)s *MeV" % config,
        )

        SelB2gammaX = self._makeB2GammaX(
            eeXLine_name + "3",
            photons=SelPhoton,
            hadrons=[
                SelKstars,
                SelPhis,
                SelLambdasLL,
                SelLambdasDD,
                SelLambdastars,
                SelK1s,
            ],
            params=config,
            masscut="ADAMASS('B+') <  %(BMassWindow)s *MeV" % config,
        )

        # 5 : Declare Lines

        SPDFilter = {
            "Code": " ( recSummary(LHCb.RecSummary.nSPDhits,'Raw/Spd/Digits') < 600 )",
            "Preambulo": [
                "from LoKiNumbers.decorators import *",
                "from LoKiCore.basic import LHCb",
            ],
        }

        self.B2eeXLine = StrippingLine(
            eeXLine_name + "Line",
            prescale=config["Bu2eeLinePrescale"],
            postscale=1,
            selection=SelB2eeX,
            RelatedInfoTools=config["RelatedInfoTools"],
            FILTER=SPDFilter,
            RequiredRawEvents=[],
            MDSTFlag=False,
        )

        self.B2eeXFromTracksLine = StrippingLine(
            eeXLine_name + "Line2",
            prescale=config["Bu2eeLine2Prescale"],
            postscale=1,
            selection=SelB2eeXFromTracks,
            RelatedInfoTools=config["RelatedInfoTools"],
            FILTER=SPDFilter,
            RequiredRawEvents=[],
            MDSTFlag=False,
        )

        self.B2mmXLine = StrippingLine(
            mmXLine_name + "Line",
            prescale=config["Bu2mmLinePrescale"],
            postscale=1,
            selection=SelB2mmX,
            RelatedInfoTools=config["RelatedInfoTools"],
            FILTER=SPDFilter,
            RequiredRawEvents=[],
            MDSTFlag=False,
        )

        self.B2meXLine = StrippingLine(
            meXLine_name + "Line",
            prescale=config["Bu2meLinePrescale"],
            postscale=1,
            selection=SelB2meX,
            RelatedInfoTools=config["RelatedInfoTools"],
            FILTER=SPDFilter,
            RequiredRawEvents=[],
            MDSTFlag=False,
        )

        self.B2meX_SSLine = StrippingLine(
            meXSSLine_name + "Line",
            prescale=config["Bu2meSSLinePrescale"],
            postscale=1,
            selection=SelB2meX_SS,
            RelatedInfoTools=config["RelatedInfoTools"],
            FILTER=SPDFilter,
            RequiredRawEvents=[],
            MDSTFlag=False,
        )

        self.B2mtXLine = StrippingLine(
            mtXLine_name + "Line",
            prescale=config["Bu2mtLinePrescale"],
            postscale=1,
            selection=SelB2mtX,
            RelatedInfoTools=config["RelatedInfoTools"],
            FILTER=SPDFilter,
            RequiredRawEvents=[],
            MDSTFlag=False,
            MaxCandidates=30,
        )

        self.B2mtX_SSLine = StrippingLine(
            mtXSSLine_name + "Line",
            prescale=config["Bu2mtSSLinePrescale"],
            postscale=1,
            selection=SelB2mtX_SS,
            RelatedInfoTools=config["RelatedInfoTools"],
            FILTER=SPDFilter,
            RequiredRawEvents=[],
            MDSTFlag=False,
            MaxCandidates=30,
        )

        self.B2gammaXLine = StrippingLine(
            eeXLine_name + "Line3",
            prescale=config["Bu2eeLine3Prescale"],
            postscale=1,
            selection=SelB2gammaX,
            RelatedInfoTools=config["RelatedInfoTools"],
            FILTER=SPDFilter,
            RequiredRawEvents=[],
            MDSTFlag=False,
        )

        # 6 : Register Lines

        self.registerLine(self.B2eeXLine)
        self.registerLine(self.B2eeXFromTracksLine)
        self.registerLine(self.B2mmXLine)
        self.registerLine(self.B2meXLine)
        self.registerLine(self.B2meX_SSLine)
        self.registerLine(self.B2mtXLine)
        self.registerLine(self.B2mtX_SSLine)
        self.registerLine(self.B2gammaXLine)

    #####################################################
    def _filterHadron(self, name, sel, params):
        """
        Filter for all hadronic final states
        """

        # requires all basic particles to have IPCHI2 > KaonIPCHI2
        # and hadron PT > KaonPT
        # need to add the ID here
        _Code = (
            "(PT > %(KaonPT)s *MeV) & "
            "(M < %(DiHadronMass)s*MeV) & "
            "((ISBASIC & (MIPCHI2DV(PRIMARY) > %(KaonIPCHI2)s)) | "
            "(NDAUGHTERS == NINTREE(ISBASIC & (MIPCHI2DV(PRIMARY) > %(KaonIPCHI2)s))))"
            % params
        )

        _Filter = FilterDesktop(Code=_Code)

        return Selection(name, Algorithm=_Filter, RequiredSelections=[sel])

    #####################################################
    def _filterDiLepton(self, name, dilepton, params, idcut=None):
        """
        Handy interface for dilepton filter
        """

        _Code = (
            "(ID=='J/psi(1S)') & "
            "(PT > %(DiLeptonPT)s *MeV) & "
            "(MM < %(UpperMass)s *MeV) & "
            "(MINTREE(ABSID<14,PT) > %(LeptonPT)s *MeV) & "
            "(MINTREE(ABSID<14,MIPCHI2DV(PRIMARY)) > %(LeptonIPCHI2)s) & "
            "(VFASPF(VCHI2/VDOF) < 9) & (BPVVDCHI2 > %(DiLeptonFDCHI2)s) & "
            "(MIPCHI2DV(PRIMARY) > %(DiLeptonIPCHI2)s)" % params
        )

        # add additional cut on PID if requested
        if idcut:
            _Code += " & " + idcut

        _Filter = FilterDesktop(Code=_Code)

        return Selection(name, Algorithm=_Filter, RequiredSelections=[dilepton])

    #####################################################
    def _filterMuTau(self, name, dilepton, params, idcut=None):
        """
        Handy interface for mutau filter
        """

        _Code = (
            "(ID=='J/psi(1S)') & "
            "(PT > %(DiLeptonPT)s *MeV) & "
            "(MM < %(UpperMass)s *MeV) & "
            "(MINTREE(ABSID<14,PT) > %(LeptonPT)s *MeV) & "
            "(MINTREE(ABSID<14,MIPCHI2DV(PRIMARY)) > %(LeptonIPCHI2)s) & "
            "(VFASPF(VCHI2/VDOF) < %(TauVCHI2DOF)s) & (BPVVDCHI2 > %(DiLeptonFDCHI2)s) & "
            "(MIPCHI2DV(PRIMARY) > %(DiLeptonIPCHI2)s)" % params
        )

        # add additional cut on PID if requested
        if idcut:
            _Code += " & " + idcut

        _Filter = FilterDesktop(Code=_Code)

        return Selection(name, Algorithm=_Filter, RequiredSelections=[dilepton])

    #####################################################
    def _filterPhotons(self, name, photons):
        """
        Filter photon conversions
        """

        _Code = "(PT > 1000*MeV) & (HASVERTEX) & (VFASPF(VCHI2/VDOF) < 9)"

        _Filter = FilterDesktop(Code=_Code)

        return Selection(name, Algorithm=_Filter, RequiredSelections=[photons])

    #####################################################
    def _makeKstarPlus(self, name, kshorts, pions, params):
        """
        Make a K*(892)+ -> KS0 pi+
        """

        _Decays = "[K*(892)+ -> KS0 pi+]cc"

        _CombinationCut = (
            "(APT > %(KaonPT)s *MeV) & "
            "(ADAMASS('K*(892)+') < %(KstarPMassWindow)s *MeV) & "
            "(ADOCACHI2CUT( %(KstarPADOCACHI2)s  , ''))" % params
        )

        _MotherCut = "(VFASPF(VCHI2) < %(KstarPVertexCHI2)s)" % params

        _KshortCut = (
            "(PT > %(KaonPT)s *MeV) & "
            "(M < %(DiHadronMass)s*MeV) & "
            "(NDAUGHTERS == NINTREE(ISBASIC & (MIPCHI2DV(PRIMARY) > %(KaonIPCHI2)s)))"
            % params
        )

        _PionCut = (
            "(PT > %(KaonPT)s *MeV) & "
            "(M < %(DiHadronMass)s*MeV) & "
            "(ISBASIC & (MIPCHI2DV(PRIMARY) > %(KaonIPCHI2)s))" % params
        )

        _Combine = CombineParticles(
            DecayDescriptor=_Decays,
            CombinationCut=_CombinationCut,
            MotherCut=_MotherCut,
        )

        _Combine.DaughtersCuts = {"KS0": _KshortCut, "pi+": _PionCut}

        return Selection(name, Algorithm=_Combine, RequiredSelections=[kshorts, pions])

    #####################################################
    def _makeK1(self, name, kaons, pions, params):
        """
        Make a K1 -> K+pi-pi+
        """

        _Decays = "[K_1(1270)+ -> K+ pi+ pi-]cc"

        # define all the cuts
        _K1Comb12Cuts = (
            "(AM > %(K1_MassWindow_Lo)s*MeV) & (AM < %(K1_MassWindow_Hi)s*MeV)" % params
        )
        _K1CombCuts = (
            "(AM > %(K1_MassWindow_Lo)s*MeV) & (AM < %(K1_MassWindow_Hi)s*MeV) & ((APT1+APT2+APT3) > %(K1_SumPTHad)s*MeV)"
            % params
        )

        _K1MotherCuts = (
            "(VFASPF(VCHI2) < %(K1_VtxChi2)s) & (SUMTREE(MIPCHI2DV(PRIMARY),((ABSID=='K+') | (ABSID=='K-') | (ABSID=='pi+') | (ABSID=='pi-')),0.0) > %(K1_SumIPChi2Had)s)"
            % params
        )
        _daughtersCuts = (
            "(TRCHI2DOF < %(Trk_Chi2)s) & (TRGHOSTPROB < %(Trk_GhostProb)s)" % params
        )

        _Combine = DaVinci__N3BodyDecays()

        _Combine.DecayDescriptor = _Decays

        _Combine.DaughtersCuts = {"K+": _daughtersCuts, "pi+": _daughtersCuts}

        _Combine.Combination12Cut = _K1Comb12Cuts
        _Combine.CombinationCut = _K1CombCuts
        _Combine.MotherCut = _K1MotherCuts

        # make and store the Selection object
        return Selection(name, Algorithm=_Combine, RequiredSelections=[kaons, pions])

    ####################################################
    def _makeMuE(self, name, params, electronid=None, muonid=None, samesign=False):
        """
        Makes MuE combinations
        """

        from StandardParticles import StdLooseMuons as Muons
        from StandardParticles import StdLooseElectrons as Electrons

        _DecayDescriptor = "[J/psi(1S) -> mu+ e-]cc"
        if samesign:
            _DecayDescriptor = "[J/psi(1S) -> mu+ e+]cc"

        _MassCut = "(AM > 100*MeV)"

        _MotherCut = "(VFASPF(VCHI2/VDOF) < 9)"

        _DaughtersCut = (
            "(PT > %(LeptonPT)s) & " "(MIPCHI2DV(PRIMARY) > %(LeptonIPCHI2)s)" % params
        )

        _Combine = CombineParticles(
            DecayDescriptor=_DecayDescriptor,
            CombinationCut=_MassCut,
            MotherCut=_MotherCut,
        )

        _MuonCut = _DaughtersCut
        _ElectronCut = _DaughtersCut

        if muonid:
            _MuonCut += "&" + muonid
        if electronid:
            _ElectronCut += "&" + electronid

        _Combine.DaughtersCuts = {"mu+": _MuonCut, "e+": _ElectronCut}

        return Selection(
            name, Algorithm=_Combine, RequiredSelections=[Muons, Electrons]
        )

    #####################################################
    def _makeMuTau(self, name, params, tauid=None, muonid=None, samesign=False):
        """
        Makes MuTau combinations
        """

        from StandardParticles import StdLooseMuons as Muons

        # from CommonParticles import StdLooseDetachedTau
        # Taus = DataOnDemand(Location = "Phys/StdLooseDetachedTau3pi/Particles")
        from CommonParticles import StdTightDetachedTau

        Taus = DataOnDemand(Location="Phys/StdTightDetachedTau3pi/Particles")

        _DecayDescriptor = "[J/psi(1S) -> mu+ tau-]cc"
        if samesign:
            _DecayDescriptor = "[J/psi(1S) -> mu+ tau+]cc"

        _MassCut = "(AM > 100*MeV)"

        _MotherCut = "(VFASPF(VCHI2/VDOF) < %(TauVCHI2DOF)s)" % params

        _DaughtersCut = (
            "(PT > %(LeptonPT)s) & " "(MIPCHI2DV(PRIMARY) > %(LeptonIPCHI2)s)" % params
        )

        _Combine = CombineParticles(
            DecayDescriptor=_DecayDescriptor,
            CombinationCut=_MassCut,
            MotherCut=_MotherCut,
        )

        _MuonCut = _DaughtersCut
        _TauCut = _DaughtersCut

        if muonid:
            _MuonCut += "&" + muonid
        if tauid:
            _TauCut += "&" + tauid

        _Combine.DaughtersCuts = {"mu+": _MuonCut, "tau+": _TauCut}

        return Selection(name, Algorithm=_Combine, RequiredSelections=[Muons, Taus])

    #####################################################
    def _makeB2LLX(
        self, name, dilepton, hadrons, params, masscut="(ADAMASS('B+')< 1500 *MeV"
    ):
        """
        CombineParticles / Selection for the B
        """

        _Decays = [
            "[ B+ -> J/psi(1S) K+ ]cc",
            "[ B+ -> J/psi(1S) pi+ ]cc",
            "[ B+ -> J/psi(1S) K*(892)+ ]cc",
            "[ B+ -> J/psi(1S) K_1(1270)+ ]cc",
            "[ B0 -> J/psi(1S) KS0 ]cc",
            "[ B0 -> J/psi(1S) K*(892)0 ]cc",
            "[ B_s0 -> J/psi(1S) phi(1020) ]cc",
            "[ Lambda_b0 -> J/psi(1S) Lambda0 ]cc",
            "[ Lambda_b0 -> J/psi(1S) Lambda(1520)0 ]cc",
        ]

        _Cut = (
            "((VFASPF(VCHI2/VDOF) < %(BVertexCHI2)s) "
            "& (BPVIPCHI2() < %(BIPCHI2)s) "
            "& (BPVDIRA > %(BDIRA)s) "
            "& (BPVVDCHI2 > %(BFlightCHI2)s))" % params
        )

        _Combine = CombineParticles(
            DecayDescriptors=_Decays, CombinationCut=masscut, MotherCut=_Cut
        )

        _Merge = MergedSelection("Merge" + name, RequiredSelections=hadrons)

        return Selection(
            name, Algorithm=_Combine, RequiredSelections=[dilepton, _Merge]
        )

    #####################################################
    def _makeB2GammaX(
        self, name, photons, hadrons, params, masscut="(ADAMASS('B+')< 1500 *MeV"
    ):
        """
        CombineParticles / Selection for the B
        """

        _Decays = [
            "[ B0   -> gamma K*(892)0 ]cc",
            "[ B_s0 -> gamma phi(1020) ]cc",
            "[ Lambda_b0 -> gamma Lambda0 ]cc",
            "[ Lambda_b0 -> gamma Lambda(1520)0 ]cc",
        ]

        _Cut = (
            "((VFASPF(VCHI2/VDOF) < %(BVertexCHI2)s) "
            "& (BPVIPCHI2() < %(BIPCHI2)s) "
            "& (BPVDIRA > %(BDIRA)s) "
            "& (BPVVDCHI2 > %(BFlightCHI2)s))" % params
        )

        _Combine = CombineParticles(
            DecayDescriptors=_Decays, CombinationCut=masscut, MotherCut=_Cut
        )

        _Merge = MergedSelection("Merge" + name, RequiredSelections=hadrons)

        return Selection(name, Algorithm=_Combine, RequiredSelections=[_Merge, photons])


#####################################################
