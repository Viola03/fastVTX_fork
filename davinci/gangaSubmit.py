# Basem khanji,basem.khanji@cern.ch: ganga script to submit tagging tuples for RunI, RunII data (both DIMUON & BHADRON streams) in one go !

# lb-run Ganga v602r3 ganga gangaSubmit.py

#
import os, re

# os.getcwd()
# sys.path.append(os.getcwd())

# running conditions
# RunI
"""
year         = [   '12'    ]#'11'     ]#,  '15'     , '16'  ]
energy       = [  '4000'   ]#'3500'   ]#,  '6500'   , '6500']
strip_v      = [   '21'    ]#'21r1'   ]#,  '24'     , '26'  ]
Reco_v       = [   '14'    ]#  '14'   ]#,  '15a'    , '16'  ]
polarity     = [   'Up'    ]#,    'Down'   ]
streams      = [ 'BHADRON.MDST']# 
#RunII
"""
year = ["18"]  # [ '17'     ]#, ] '16'   ]  '15'
energy = ["6500"]  # [ '6500'   ]#, ] '6500' ]  '6500'
strip_v = ["34"]  # [ '29r2'   ]#, ] '28r1' ]  '24r1'
Reco_v = ["18"]  # [ '17'     ]#, ] '16'   ]  '15a'
# polarity     = [ 'Up'   , 'Down' ]
polarity = ["Up"]
streams = ["BHADRONCOMPLETEEVENT.DST"]  # 'DIMUON'

# make the paths list
job_setting = {}
List_Of_Paths = []

for stm in streams:
    for pol in polarity:
        for i in range(len(year)):
            PATH_name = (
                "/LHCb/Collision"
                + year[i]
                + "/Beam"
                + energy[i]
                + "GeV-VeloClosed-Mag"
                + pol
                + "/Real Data/Reco"
                + Reco_v[i]
                + "/Stripping"
                + strip_v[i]
                + "/90000000/"
                + stm
            )
            print(PATH_name)
            job_name = (
                "20"
                + year[i]
                + "_Reco"
                + Reco_v[i]
                + "Strip"
                + strip_v[i]
                + "_"
                + pol
                + "_"
                + stm
            )
            job_setting[job_name] = PATH_name
            List_Of_Paths.append(PATH_name)

print("========================================")
print("Filled the list of PATHS for ganga jobs")
print("========================================")
print(job_setting)


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

    # myApp.directory = '/afs/cern.ch/work/b/bkhanji/DaVinciDev_v44r3'
    # myApp.directory = '/afs/cern.ch/work/b/bkhanji/DaVinciDev_v42r3'
    myApp.platform = "x86_64-slc6-gcc62-opt"
    # myApp.platform = 'x86_64-slc6-gcc49-opt'
    myApp.options = ["./Data/B2DD_TupleMaker_Data.py"]
    # myApp.options = ['./Data/B2DD_TupleMaker_Data_RunI.py' ]
    # Choose PBS backend and specify walltime
    bck = Dirac()
    # bck = Local()
    # bck = Interactive()

    # Split into subjobs, defining maximum number of input files to analyse
    # and number of input files per subjob
    splitter = SplitByFiles()
    splitter.ignoremissing = True
    splitter.maxFiles = -1
    splitter.filesPerJob = 5

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
        + "from Configurables import CondDB                      ; "
        + "CondDB( LatestGlobalTagByDataType = "
        + Year
        + ")     ; "
        + "DaVinci().DataType      =   "
        + Year
        + "              ; "
    )

    print("Create job for thee jobs: ", job.name)
    # job.inputdata  = dataset
    # job.inputdata  = [dataset[0]]
    job.inputdata = dataset[:50]
    # job.outputfiles= [LocalFile(namePattern='*.root') ] # keep my Tuples on grid element (retrive manually)
    # This throws the files on the grid personall space
    job.outputfiles = [
        DiracFile(namePattern="*.root"),
        LocalFile("summary.xml"),
    ]  # keep my Tuples on grid element (retrive manually)
    # job.outputfiles= [LocalFile(namePattern='*.root') , LocalFile('summary.xml') ] # keep my Tuples on grid element (retrive manually)
    jobs.parallel_submit = True
    job.submit()
    print("======================================")
    print("job: ", job.name + " submitted")
    print("======================================")
print(" Jobs submitted .... bye ")
