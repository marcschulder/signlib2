# import sys
import base64
import os


def du(im):
    imgfile = open(im, 'rb').read()
    b64img = base64.b64encode(imgfile)
    file_name = os.path.splitext(im)
    fname = file_name[0]

    fext = file_name[1]

    b64imgfile = open(fname + fext + '.txt', 'w')
    for line in b64img:
        b64imgfile.write(line)
    print(fname)
    print(fext)
    print('done')
