import pandas as pd
from upscaling import calc_footprint_FFP_climatology as myfootprint
import numpy as np
import matplotlib.pyplot as plt
import os
import re


excel_path = r'J:\Arou\2016\Arou_ustar_zml_h.xlsx'


output_path = r'J:\Arou\2016\xyfclimfr2016'


os.makedirs(output_path, exist_ok=True)


df = pd.read_excel(excel_path)


domaint = [-3000., 3000., -3000., 3000.]
nxt = 6000
rst = [90.]


for index, row in df.iterrows():

    if row['name'] < 2162:
        continue

    zmt = row['zm']
    umeant = [row['umean']]
    ht = [row['h']]
    olt = [row['ol']]
    sigmavt = [row['sigmav']]
    ustart = [row['u_star']]
    wind_dirt = [row['wind_dir']]
    name_suffix = row['name']


    try:
        FFP = myfootprint.FFP_climatology(zm=zmt, z0=None, umean=umeant, h=ht, ol=olt,
                                          sigmav=sigmavt, ustar=ustart, wind_dir=wind_dirt,
                                          domain=domaint, nx=nxt, rs=rst, smooth_data=1)
    except Exception as e:
        print(f"Error calculating FFP for row {index}: {e}")
        continue


    try:
        fclim_2d = FFP.fclim_2d
        x_2d = FFP.x_2d
        y_2d = FFP.y_2d
        fr = FFP.fr
    except AttributeError:
        fclim_2d = FFP['fclim_2d']
        x_2d = FFP['x_2d']
        y_2d = FFP['y_2d']
        fr = FFP['fr']


    assert isinstance(fclim_2d, np.ndarray), "fclim_2d should be a NumPy array."
    assert isinstance(x_2d, np.ndarray), "x_2d should be a NumPy array."
    assert isinstance(y_2d, np.ndarray), "y_2d should be a NumPy array."

    fr_value = float(re.search(r'\[(.*)\]', str(fr)).group(1)) if isinstance(fr, str) else fr


    df_fclim_2d = pd.DataFrame(fclim_2d)
    df_x_2d = pd.DataFrame(x_2d)
    df_y_2d = pd.DataFrame(y_2d)
    df_fr = pd.DataFrame({'fr': [fr_value]})


    output_file = os.path.join(output_path, f'output_{name_suffix}.xlsx')


    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')


    df_fclim_2d.to_excel(writer, sheet_name='fclim_2d', index=False)
    df_x_2d.to_excel(writer, sheet_name='x_2d', index=False)
    df_y_2d.to_excel(writer, sheet_name='y_2d', index=False)
    df_fr.to_excel(writer, sheet_name='fr', index=False)


    writer.save()
    plt.close()
    print(f"fclim_2d, x_2d, y_2d, and fr for name {name_suffix} have been saved to {output_file}.")
