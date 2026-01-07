import pandas as pd
from upscaling import calc_footprint_FFP_climatology as myfootprint
import numpy as np
import matplotlib.pyplot as plt
import os
import re  # 引入正则表达式模块

# CSV文件路径
excel_path = r'J:\Arou\2016\Arou_ustar_zml_h.xlsx'  # 替换为您的CSV文件路径

# 输出路径
output_path = r'J:\Arou\2016\xyfclimfr2016'

# 确保输出路径存在
os.makedirs(output_path, exist_ok=True)

# 读取CSV文件
df = pd.read_excel(excel_path)

# 固定参数
domaint = [-3000., 3000., -3000., 3000.]
nxt = 6000
rst = [90.]

# 循环遍历DataFrame的每一行
for index, row in df.iterrows():
    # 仅处理 name 列大于或等于 9235 的行
    if row['name'] < 2162:
        continue  # 跳过这一行
    # 提取当前行的参数值
    zmt = row['zm']
    umeant = [row['umean']]
    ht = [row['h']]
    olt = [row['ol']]
    sigmavt = [row['sigmav']]
    ustart = [row['u_star']]
    wind_dirt = [row['wind_dir']]
    name_suffix = row['name']  # 假设这一列是名称后缀

    # 计算FFP
    try:
        FFP = myfootprint.FFP_climatology(zm=zmt, z0=None, umean=umeant, h=ht, ol=olt,
                                          sigmav=sigmavt, ustar=ustart, wind_dir=wind_dirt,
                                          domain=domaint, nx=nxt, rs=rst, smooth_data=1)
    except Exception as e:
        print(f"Error calculating FFP for row {index}: {e}")
        continue

    # 获取 fclim_2d、x_2d 和 y_2d 的值
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

    # 检查 fclim_2d、x_2d 和 y_2d 的类型
    assert isinstance(fclim_2d, np.ndarray), "fclim_2d should be a NumPy array."
    assert isinstance(x_2d, np.ndarray), "x_2d should be a NumPy array."
    assert isinstance(y_2d, np.ndarray), "y_2d should be a NumPy array."

    # 处理 fr 值，去掉方括号
    fr_value = float(re.search(r'\[(.*)\]', str(fr)).group(1)) if isinstance(fr, str) else fr

    # 创建 DataFrame
    df_fclim_2d = pd.DataFrame(fclim_2d)
    df_x_2d = pd.DataFrame(x_2d)
    df_y_2d = pd.DataFrame(y_2d)
    df_fr = pd.DataFrame({'fr': [fr_value]})  # 使用处理后的 fr_value

    # 输出文件路径
    output_file = os.path.join(output_path, f'output_{name_suffix}.xlsx')

    # 创建 ExcelWriter 对象
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

    # 将 DataFrame 写入 Excel 文件的不同工作表
    df_fclim_2d.to_excel(writer, sheet_name='fclim_2d', index=False)
    df_x_2d.to_excel(writer, sheet_name='x_2d', index=False)
    df_y_2d.to_excel(writer, sheet_name='y_2d', index=False)
    df_fr.to_excel(writer, sheet_name='fr', index=False)

    # 保存 Excel 文件
    writer.save()
    plt.close()
    print(f"fclim_2d, x_2d, y_2d, and fr for name {name_suffix} have been saved to {output_file}.")
