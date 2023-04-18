import os, json
import Data.IO as dio
from Volume.Selection import selectPoints
import pandas as pd
import numpy as np

def extractPointsFromPatient(patient, r=20, nbPoints=100, outfile="points.csv", randomPoints=False):
    os.chdir(patient)
    print('Load volume from disk: '+patient)
    with open("config.json", 'r') as f:
        d = json.load(f)
    vol, vox2met = dio.read_nii_from_file(d['noskull volume'])
    print(f'Extracting points')
    fPoints = dio.read_points_from_csv(d['pts aneurysm'])
    fp = dio.points_to_spheres(fPoints)[:,:-1] # drop centers
    q = None
    if randomPoints:
        minThreshold, maxThreshold = np.percentile(vol[vol>0], 0), np.percentile(vol[vol>0], 100)
        print(f'\Random Points ', end='')
        p=selectPoints(vol, vox2met, thresLow=minThreshold, thresHigh=maxThreshold, r=r,
                             fPoints=fp, nbPoints=nbPoints, extractType='Parenchyma')
        print(f'{len(p)} random points')
        title = 'Random'
    else:
        T= np.percentile(vol[vol>0], 95)
        minThreshold = np.percentile(vol[vol>0], 50)
        maxThreshold = np.percentile(vol[vol>0], 90)
        p = selectPoints(vol, vox2met, thresLow=T, r=r, fPoints=fp, nbPoints=nbPoints, extractType='Vessels')
        title = 'Vessel'
        print(f'{len(p)} vessel points')
        q=selectPoints(vol,vox2met,thresLow=minThreshold,thresHigh=maxThreshold,r=r,fPoints=np.vstack((fp,p)),
                                nbPoints=nbPoints,extractType='Parenchyma')
        print(f'{len(q)} Parenchyma points')

    # export to csv using pandas
    #print(f'Exporting to CSV')
    ps = pd.DataFrame(p, columns=list('xyz'))
    ps['type'] = pd.Categorical([title] * len(p))
    if q is not None:
        qs = pd.DataFrame(q, columns=list('xyz'))
        qs['type']=pd.Categorical(['Parenchyma']*len(q))
        points=ps.append(qs)
    else:
        points = ps
    points.to_csv(outfile)
    
def extractPoints(dataPath, r = 20, nbPoints=100, outfile="points.csv", randomPoints = False):
    patients = dio.fetch_patient_dirs(dataPath)
    for patient in patients:
        extractPointsFromPatient(patient, r=r, nbPoints=nbPoints, outfile=outfile, randomPoints=randomPoints)

