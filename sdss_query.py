#! /usr/bin/env python3

#Copyright 2019 Kyle Steckler

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, merge, 
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons 
# to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or 
# substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import SciServer
from SciServer import Authentication, SkyServer, CasJobs, SkyQuery
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from getpass import getpass
import pdb
import sys

def create_datafiles(n_galaxies = 200,galaxy_type = 'both', lower_z_limit=0.1, upper_z_limit = 0.3, lower_flux_limit = 50, upper_flux_limit = 500, data_release = 'DR15', image_data = True, image_scale_factor = 0.01):
    """
    Explain what func does and all parameters

    """
    auth_login = input("Username: ")
    auth_pass = getpass()

    try:
        Authentication.login(auth_login,auth_pass)
        print('Login Successful: Getting Tables...')
    except Exception:
        sys.exit("Login Failed. (May be locked out if excessive attempts are made)")
    
    CasJobs.getTables(context='MyDB')

    try:
        SkyQuery.dropTable('galaxies', datasetName='MyDB')
        
    except Exception:
        print("WARNRING: Unable to drop table (it may not exist yet)")

    SQL_Query = f"SELECT TOP {n_galaxies} g.objid, g.ra, g.dec,g.petroR90_g, g.petroFlux_g, "
    SQL_Query += "g.specObjID, g.lnLDeV_g, s.z FROM Galaxy as g "
    SQL_Query += "JOIN SpecObjAll as s ON g.specObjID = s.specObjID "
    SQL_Query += "INTO MyDB.galaxies "
    SQL_Query += f"WHERE s.z BETWEEN {lower_z_limit} AND {upper_z_limit} "
    SQL_Query += f"AND (g.petroFlux_g BETWEEN {lower_flux_limit} AND {upper_flux_limit}) "
    SQL_Query += f"AND g.type=3 AND clean=1 "

    if galaxy_type == 'both':
        SQL_Query += "AND (g.lnLDeV_r > g.lnLExp_r + 0.1 AND g.lnLExp_r > -999.0 AND g.lnLDeV_g > -999.0) "
        SQL_Query += "OR (g.lnLDeV_g < -2000.0 AND g.lnLDeV_g + 0.1 < g.lnLExp_g );"
    elif galaxy_type == 'spiral':
        SQL_Query += "AND (g.lnLDeV_g < -1000.0 AND g.lnLDeV_g < g.lnLExp_g); "
    elif galaxy_type == 'elliptical':
        SQL_Query += "AND g.lnLDeV_r > g.lnLExp_r + 0.1 AND g.lnLExp_r > -999.0 AND g.lnLDeV_g > -999.0;"
    else:
        raise Exception("Invalid Galaxy Type: valid options are both, spiral, elliptical")
    
    print('Querying Database...') 
    job_id = CasJobs.submitJob(sql=SQL_Query, context=data_release)
    state = CasJobs.waitForJob(job_id,verbose=True)
    del state

    df = SkyQuery.getTable('galaxies',datasetName='MyDB',top=n_galaxies)
    if len(df) != n_galaxies:
        print(f"Was only able to find {len(df)} galaxies with current search parameters")


    ## 0: 'probably spiral', 1: 'probably elliptical'
    df['Classification'] = df['lnLDeV_g'].apply(lambda x: 0 if x < -1000.0 else 1)
    print(f"DISTRIBUTION: \n Spirals: {len(df[df['Classification'] == 0])} \n Ellipticals: {len(df[df['Classification'] == 1])}\n")
    
    if image_data:
        print("Grabbing Image Data ...")
        images = get_image_data(df, scaling_factor = image_scale_factor)
        np.save('galaxy_images', images)
        print("File Created: galaxy_images.npy")


    #df.to_csv('galaxy_data.csv', index=False)
    galaxy_labels = np.array(df['Classification'])
    galaxy_labels.to_csv('galaxy_labels.npy')
    print("File Created: galaxy_labels.npy.csv")
       


def get_image_data(results, scaling_factor = 0.01):
    # RA, DEC, RADIUS OF 90% FLUX 
    RA = np.array(results['ra'])
    DEC = np.array(results['dec'])
    petroR90 = np.array(results['petroR90_g'])

    # Scale for images in arcsec/pixel
    scaler = scaling_factor * petroR90
        
    data = []
    count = 0
    for ra,dec,scale in zip(RA, DEC, scaler):
        print(f"{count}/{len(RA)} Images Done" )
        data.append(SkyServer.getJpegImgCutout(ra,dec, scale=scale))
        count += 1

    return np.array(data)













