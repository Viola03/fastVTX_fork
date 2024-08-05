import os, re

year = ["18"]
energy = ["6500"]
strip_v = ["34"]
Reco_v = ["18"]
polarity = ["Down"]
streams = ["ALLSTREAMS.DST"]

job_setting = {}
List_Of_Paths = []
i = 0
# inclusive B
# PATH_name = "/MC/2018/Beam6500GeV-2018-MagDown-Nu1.6-25ns-Pythia8/Sim09k/Trig0x617d18a4/Reco18/Turbo05-WithTurcal/Stripping34NoPrescalingFlagged/10000023/ALLSTREAMS.DST"
# Kee
# decay_name = "Kee"
# PATH_name = "/MC/2018/Beam6500GeV-2018-MagDown-Nu1.6-25ns-Pythia8/Sim09g/Trig0x617d18a4/Reco18/Turbo05-WithTurcal/Stripping34NoPrescalingFlagged/12123003/ALLSTREAMS.DST"
# K*ee
# decay_name = "Kstee"
# PATH_name = "/MC/2015/Beam6500GeV-2015-MagUp-Nu1.6-25ns-Pythia8/Sim09i/Trig0x411400a2/Reco15a/Turbo02/Stripping24r1NoPrescalingFlagged/11124002/ALLSTREAMS.DST"
# BuD0enuKenu
# decay_name = "BuD0enuKenu" # THIS HAS high q2 CUT!!!
# PATH_name = "/MC/2018/Beam6500GeV-2018-MagDown-Nu1.6-25ns-Pythia8/Sim09k/Trig0x617d18a4/Reco18/Turbo05-WithTurcal/Stripping34NoPrescalingFlagged/12583023/ALLSTREAMS.DST"
# decay_name = "BuD0piKenu" 
# PATH_name = "/MC/2018/Beam6500GeV-2018-MagDown-Nu1.6-25ns-Pythia8/Sim09m/Trig0x617d18a4/Reco18/Turbo05-WithTurcal/Stripping34NoPrescalingFlagged/12583040/ALLSTREAMS.DST"
decay_name = "Kmumu" 
PATH_name = "/MC/2018/Beam6500GeV-2018-MagUp-Nu1.6-25ns-Pythia8/Sim09g/Trig0x617d18a4/Reco18/Turbo05-WithTurcal/Stripping34NoPrescalingFlagged/12113002/ALLSTREAMS.DST"

print(PATH_name)
job_name = (
    f"{decay_name}_20"
    + year[i]
    + "_Reco"
    + Reco_v[i]
    + "Strip"
    + strip_v[i]
    + "_"
    + polarity[i]
    + "_"
    + streams[i]
)
job_setting[job_name] = PATH_name
List_Of_Paths.append(PATH_name)

for job_name, path_dict in job_setting.items():
    print("======================================")
    print("Sumbitting a new job ...")
    print(path_dict, ",", job_name)

    bk_query = BKQuery(path=path_dict)
    dataset = bk_query.getDataset()

    try:
        myApp = prepareGaudiExec("DaVinci", "v44r3", myPath=".")
    except:
        myApp = GaudiExec()
        myApp.directory = "./DaVinciDev_v44r3"

    # myApp.platform = "x86_64-slc6-gcc62-opt"
    myApp.platform = "x86_64+avx2+fma-centos7-gcc62-opt"
    # myApp.options = ["./davinci_intermediates.py", "./print_something.py"]
    myApp.options = ["./davinci_intermediates.py"]

    bck = Dirac()
    # bck = Condor()
    # bck = Local()
    # bck = Interactive()

    splitter = SplitByFiles()
    splitter.ignoremissing = True
    splitter.maxFiles = -1
    splitter.filesPerJob = 3

    job = Job(name=job_name, comment=job_name, backend=bck, splitter=splitter)
    Year = (
        bool("2011" in job_name) * ' "2011" '
        + bool("2012" in job_name) * ' "2012" '
        + bool("2015" in job_name) * ' "2015"  '
        + bool("2016" in job_name) * ' "2016"  '
        + bool("2017" in job_name) * ' "2017" '
        + bool("2018" in job_name) * ' "2018" '
    )
    job.do_auto_resubmit = False
    job.application = myApp

    job.application.extraOpts = (
        "from Configurables import DaVinci                     ; "
        + 'DaVinci().TupleFile     = "DTT_'
        + job_name
        + '.root"  ; '
        + "DaVinci().EvtMax        =              -1             ; "
        # + "DaVinci().EvtMax        =              25             ; "
        + "from Configurables import CondDB                      ; "
        + "CondDB( LatestGlobalTagByDataType = "
        + Year
        + ")     ; "
        + "DaVinci().DataType      =   "
        + Year
        + "              ; "
    )

    print("Create job for thee jobs: ", job.name)
    job.inputdata  = dataset
    # job.inputdata = [dataset[0]]
    # job.inputdata = dataset[:5]

    # This throws the files on the grid personall space
    job.outputfiles = [
        # DiracFile(namePattern="*.root"),
        LocalFile(namePattern="DTT*.root"),
        LocalFile("summary.xml"),
    ]  # keep my Tuples on grid element (retrive manually)
    # job.outputfiles= [LocalFile(namePattern='*.root') , LocalFile('summary.xml') ] # keep my Tuples on grid element (retrive manually)
    jobs.parallel_submit = True
    job.submit()
    print("======================================")
    print("job: ", job.name + " submitted")
    print("======================================")

print(" Jobs submitted .... bye ")
