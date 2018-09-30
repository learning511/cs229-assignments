import os

def getresults(resultFolder, imgFolder):

    """
    Parameters
    ----------
    resultFolder: results directory
    imgFolder: image directory
    """
    # Windows
    # cmd = 'cmd.exe /k tesseract.exe ' + img + 'result -l chi_sim+eng'

    # OS X
    # cmd = 'tesseract ' + img + ' result -l chi_sim'

    imgList = os.listdir(imgFolder)
    print(imgList)
    os.chdir(imgFolder)
    for i in range(0, len(imgList)):
        resultFile = ' %sresult-%s' % (resultFolder, imgList[i].split('.')[0])
        #cmd = 'tesseract ' + imgList[i] + resultFile + ' -l chi_sim'
        print(resultFile)
        cmd = 'cmd.exe /k tesseract.exe ' + imgList[i] + ' result -l chi_sim+eng'
        print(cmd)
        os.popen(cmd)
        print('finished!')

if __name__ == "__main__":
    getresults('F:\devtools\Tesseract-OCR\output', 'F:\devtools\Tesseract-OCR\input');