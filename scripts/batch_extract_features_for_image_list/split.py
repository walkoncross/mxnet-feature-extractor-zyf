import sys
import os
import argparse

def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--division', type=int, default=8,help='image list division number')
    parser.add_argument('--save-dir', type=str, default='./filelist',help='where to save divided filelist')
    parser.add_argument('--file-list',type=str, help='where is the filelist')
    parser.add_argument('--file-prefix',type=str,default='split', help='file prefix')
    return parser.parse_args(argv)


if __name__ == '__main__':
    argv=parse_args(sys.argv[1:])
    lines = ''
    filelist=argv.file_list      # 'megaface_aligned_face_filelist.txt'
    division=argv.division
    save_dir=argv.save_dir
    file_prefix=argv.file_prefix
   
    with open(filelist,'r') as f:
        lines = f.readlines()
    count = len(lines)
    f.close()
    for i in range(division):
	split_name=save_dir+"/"+file_prefix+"_"+str(i)+".txt"
        fout=open(split_name,"w")
	fout.writelines(lines[i* count / division:(i+1) * count / division])
        fout.close()
	print split_name

'''
    f0 = open('megaface2_aligned_face_filelist-split0.txt','w')
    f1 = open('megaface2_aligned_face_filelist-split1.txt', 'w')
    f2 = open('megaface2_aligned_face_filelist-split2.txt', 'w')
    f3 = open('megaface2_aligned_face_filelist-split3.txt', 'w')
    f4 = open('megaface2_aligned_face_filelist-split4.txt', 'w')
    f5 = open('megaface2_aligned_face_filelist-split5.txt', 'w')
    f6 = open('megaface2_aligned_face_filelist-split6.txt', 'w')
    f7 = open('megaface2_aligned_face_filelist-split7.txt', 'w')

    f0.writelines(lines[0:count/8])
    f1.writelines(lines[count/8: 2*count/8])
    f2.writelines(lines[2*count/8: 3*count/8])
    f3.writelines(lines[3*count/8: 4*count/8])
    f4.writelines(lines[4*count/8: 5*count/8])
    f5.writelines(lines[5*count/8: 6*count/8])
    f6.writelines(lines[6*count/8: 7*count/8])
    f7.writelines(lines[7*count/8: count])
   
    f0.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    f7.close()
'''
