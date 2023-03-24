
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from pathlib import Path
import os
from PIL import Image
def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

# 读取图像到列表中
cycle = sorted(list_full_paths('./metrics/plots_/cycle'), reverse=True)
DCL =sorted(list_full_paths('./metrics/plots_/DCLGAN'), reverse=True)
CUT =sorted(list_full_paths('./metrics/plots_/CUT'), reverse=True)
takeoff = sorted(list_full_paths('./metrics/plots_/take_off_glasses'), reverse=True)
ori = sorted(list_full_paths('./metrics/plots_/GlassesExamples'), reverse=True)
ours = sorted(list_full_paths('./metrics/plots_/Ours'), reverse=True)
# ori = os.listdir('./metrics/plots_/cycle')
print(len(ori),ori)

# plt.figure(figsize=(4, 9), dpi=300)
LEN=len(ori)
fname='image_path[0][0]'
idx=1
plt.figure( dpi=500)

# for i in range(1, LEN+1):
for row, j in enumerate([cycle, DCL, CUT, takeoff, ours, ori]):
    for cal, k in enumerate (j):
        if cal == 6:
            continue
        
            # match cal:
            #     case 1:
            #         plt.ylabel('Input')
        # plt.ylabel('Sample {}'.format(num))
        im = Image.open(k)
        im = im.resize((512, 512))
        ax = plt.subplot(6, 8-1, idx)
        plt.imshow(im)
        idx+=1
        if row == len(j)-1:
            plt.xlabel('Sample {}'.format(cal+1))
        if cal ==0:
            match row:
                case 0:
                    plt.ylabel('CycleGAN')
                case 1:
                    plt.ylabel('DCLGAN')
                case 2:
                    plt.ylabel('CUT')
                case 3:
                    plt.ylabel('P.E.R.N')
                case 4:
                    plt.ylabel('Ours')
                case 5:
                    plt.ylabel('Input')
        plt.xticks([])
        plt.yticks([])

        # 去除黑框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # idx+=1
    # if image_path[i-1][0] == fname[0]:
    #     pass

    
    

        
    # elif image_path[i-1].endswith('512_gmask.png'):
    #     plt.xlabel('Glass Mask')
    # elif image_path[i-1].endswith('512_smask.png'):
    #     plt.xlabel('Shadow Mask')
    # 去除坐标轴


# 设置各个子图间间距
# plt.subplots_adjust(left=0.10, top=0.88, right=0.5, bottom=0.08, wspace=0.00, hspace=0.00)
plt.subplots_adjust(left=0.2, top=0.9, right=0.8, bottom=0.1, wspace=0.00, hspace=0.00)
# plt.tight_layout()
# plt.subplots_adjust(left=0.10, top=0.88, right=0.65, bottom=0.08, wspace=0.02, hspace=0.02)
plt.savefig('res.png')






# for i in range(1, LEN+1):
#     im = Image.open('./images/'+image_path[i-1])
#     im = im.resize((512, 512))
#     ax = plt.subplot(3, 4, i)
#     plt.imshow(im)
#     # plt.colorbar()
#     if image_path[i-1][0] == fname[0]:
#         pass
#     else:
        
#         # plt.ylabel('Sample {}'.format(idx))
#         fname = image_path[i-1]
#         plt.ylabel('Sample {}'.format(idx))
#         idx+=1
#     if image_path[i-1].endswith('512.png'):
#         plt.xlabel('Input')
#     elif image_path[i-1].endswith('512_res.png'):
#         plt.xlabel('Output')
#     elif image_path[i-1].endswith('512_gmask.png'):
#         plt.xlabel('Glass Mask')
#     elif image_path[i-1].endswith('512_smask.png'):
#         plt.xlabel('Shadow Mask')
#     # 去除坐标轴
#     plt.xticks([])
#     plt.yticks([])

#     # 去除黑框
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)

# # 设置各个子图间间距
# # plt.subplots_adjust(left=0.10, top=0.88, right=0.5, bottom=0.08, wspace=0.00, hspace=0.00)
# plt.subplots_adjust(left=0.1, top=1.0, right=1.0, bottom=0.1, wspace=0.00, hspace=0.00)

# # plt.subplots_adjust(left=0.10, top=0.88, right=0.65, bottom=0.08, wspace=0.02, hspace=0.02)
# plt.savefig('res.png')
