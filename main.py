from utilities.parserUtils import *
from utilities.customUtils import *
from utilities.aestheticUtils import *
from dataTools.processDataset import *
from dataTools.patchExtractor import *
from mainModule.DFSMDNet import *

if __name__ == "__main__":


    options = mainParser(sys.argv[1:])
    if len(sys.argv) == 1:
        customPrint("Invalid option(s) selected! To get help, execute script with -h flag.")
        exit()

    if options.conf:
        configCreator()

    config = configReader()

    if options.epoch:
        config=updateConfig(entity='epoch', value=options.epoch)
    if options.batch:
        config=updateConfig(entity='batchSize', value=options.batch)
    if options.manualUpdate:
        config=manualUpdateEntity()
    if options.modelSummary:
        DFSMDNet(config).modelSummary()
    if options.train:
        DFSMDNet(config).modelTraining(dataSamples=options.dataSamples)
    if options.retrain:
        DFSMDNet(config).modelTraining(resumeTraning=True, dataSamples=options.dataSamples)
    if options.inference:
        noiseSigmaSet = None
        if options.noiseSigma:
            noiseSigmaSet = options.noiseSigma.split(',')
            noiseSigmaSet = list(map(int, noiseSigmaSet))
        DFSMDNet(config).modelInference(testImagesPath=options.sourceDir, outputDir=options.resultDir, noiseSet=noiseSigmaSet)
    if options.overFitTest:
        DFSMDNet(config).modelTraining(overFitTest=True)
    
        
        
        
            


