from HDR import HDR
from util.config import ModelConf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
if __name__ == '__main__':


    import time
    s = time.time()
    #Register your model here and add the conf file into the config directory

    try:
        conf = ModelConf('./HGHDA.conf')
    except KeyError:
        print('wrong num!')
        exit(-1)
    recSys = HDR(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
