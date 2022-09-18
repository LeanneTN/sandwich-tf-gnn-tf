# ******************************************************************************
# *            Copyright (c) 2021 CSS518, chenhao<2411889319@qq.com>           *
# *                                                                            *
# *    Permission is hereby granted, free of charge, to any person obtaining   *
# *    a copy of this software and associated documentation files (the         *
# *    "Software"), to deal in the Software without restriction, including     *
# *    without limitation the rights to use, copy, modify, merge, publish,     *
# *    distribute, sublicense, and/or sell copies of the Software, and to      *
# *    permit persons to whom the Software is furnished to do so, subject to   *
# *    the following conditions:                                               *
# *                                                                            *
# *    The above copyright notice and this permission notice shall be          *
# *    included in all copies or substantial portions of the Software.         *
# *                                                                            *
# *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,         *
# *    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF      *
# *    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                   *
# *    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE  *
# *    LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION  *
# *    OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION   *
# *    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.         *
# ******************************************************************************


import os
import time
import re

dockerName2student = {
    'rong2': '陈榕',
    'chengyi': '程弋',
    'hans': '李智涵',
    'swallow': '陈昊',
    'ouyangshuying': '欧阳澍崟',
    'zhangh2': '张欢',
    'zhou2': '周聪',
    'Columbus': '官天真',
    'shiry': '施如意',
    'zhengjianbo': '郑剑波',
    'Mikeneko': '张越',
    'chengzi_cuda10.0': '杨海洋',
    'pikachu_cuda10': '杨浩',
    'yankeeZ_cuda10.1_jupyter_new': '张洋奇',
    'LeslieGe_cuda10.0': '葛凡',
    'xiaowanzhi': '肖万志',
}


dic_studentid_to_name={
    
}


def print_html_header():
    print("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GPU使用情况</title>
</head>
<body>
<main style="">
<h1>CSS518高性能工作站使用情况概览</h1>
注意：数据每5分钟更新一次    
<pre>
    """, file=f)

def print_html_footer():
    print("""
</pre>
</main>
</body>
</html>
    """, file=f)



def checknumber(s):
    for c in s:
        if '0'>c or c>'9':
            return False
    return True

def print_html_memory_usage():
    #检查是否为全数字

    res_str=''
    res_str+='+--------------------------------------------------------+\n'
    res_str+='|   student id              姓名              disk used  |\n'
    #dire是学号，sttr[0]是使用空间
    directory=os.listdir('/gpudata')
    for dire in directory:
        if checknumber(dire):
            sstr=os.popen("du -sh /gpudata/{} 2>/dev/null".format(dire)).read()
            sstr=re.findall(r'(.*?)\s.*?',sstr)
            if dire in dic_studentid_to_name.keys():
                res_str+='| {:>12}              {:<11}{:>14}    |\n'.format(dire,dic_studentid_to_name[dire],sstr[0])
            else:
                res_str+='| {:>12}              {:<11}{:>14}    |\n'.format(dire,"Unknown",sstr[0])
    res_str+='+--------------------------------------------------------+'

    # print(res_str,file=f)
    return res_str


def show_nvidia_smi():
    process_id = os.popen(r"nvidia-smi")
    output = process_id.read()
    print(output, file=f)

def show_gpu_usage():
    process_id = os.popen(r"nvidia-smi | grep -A 25 ========================================| awk '{print $5}'")
    process_id = process_id.read()
    process_id = process_id.strip().split()

    gpu_id = os.popen(r"nvidia-smi | grep -A 25 ========================================| awk '{print $2}'")
    gpu_id = gpu_id.read()
    gpu_id = gpu_id.strip().split()

    container_names = os.popen(r"docker ps | grep Up|awk '{print $NF}'")
    container_names = container_names.read()
    container_names = container_names.strip().split()
    # print(process_id,gpu_id,container_names)
    for name in container_names:
        for gpu, pid in zip(gpu_id, process_id):
            s = os.popen("docker top " + name + " | grep " + '" ' +pid+' "')
            s = s.read()
            if s != '':
                if name in dockerName2student:
                    student_name = dockerName2student[name]
                else:
                    student_name = 'unknown real name, docker name is {}'.format(name)
                print("{{{}}}	{}--------->{}".format(gpu, pid, student_name), file=f)
            # pri=os.popen("if [ -n \"{}\" ];then echo \"{{{}}}      {}------->{}\";fi".format(s,gpu,pid,name))
            # print(pri.read())

    # s=`docker top ${i} | grep ${j}`; if [ -n "${s}" ];
    # then echo $j"------->"$i &&echo ${s}

if __name__ == '__main__':
    count=0
    memory=""
    print("begin")
    while True:
        with open('./gpus.html.bak', 'w', encoding='utf-8') as f:
            print_html_header()
            show_nvidia_smi()
            if count==0:
                memory=print_html_memory_usage()
            if count==100:
                count=-1

            show_gpu_usage()
            print(memory,file=f)
            #print_html_memory_usage()
            print_html_footer()
        count+=1
        os.popen(r'cp gpus.html.bak gpus.html')
        print('sleep')
        # 每5分钟跑一次
        time.sleep(60 * 3)

print("file")