import glob,os

#Confirmed misalignments in 
misalignments = [10034,10045,10172,183]
dataset_dir = './dataset'

for ids in misalignments:
    files = glob.glob(dataset_dir+'/*/*/%05d_*.ARW'%ids)
    for file in files:
        original = file
        path = os.path.dirname(file)
        file = os.path.basename(file)
        file = list(file)
        file[0] = '5'
        file = "".join(file)
        file = path+'/'+file
        os.rename(original,file)