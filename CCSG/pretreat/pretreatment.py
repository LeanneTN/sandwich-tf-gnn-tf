import sys
sys.path.insert(0,"..")
import javalang

import javalang.tokenizer as tokenizer
import re
from javalang.parser import Parser
replace_type=[tokenizer.Null,tokenizer.String,tokenizer.Identifier,tokenizer.Boolean,tokenizer.FloatingPoint,tokenizer.Integer,tokenizer.DecimalInteger]
not_replace_type=[tokenizer.Operator,tokenizer.Modifier,tokenizer.Separator
                  ,tokenizer.Annotation
                  ]
import javalang.tree
import random
import pickle
import argparse

main_type_in=[javalang.tree.IfStatement,javalang.tree.ForStatement,javalang.tree.MethodInvocation,
              javalang.tree.FormalParameter,javalang.tree.CatchClauseParameter,javalang.tree.SuperMethodInvocation,
              javalang.tree.SwitchStatement,javalang.tree.SynchronizedStatement,javalang.tree.ClassCreator,
              javalang.tree.WhileStatement,javalang.tree.DoStatement,javalang.tree.ThrowStatement,
              javalang.tree.SuperConstructorInvocation
              ]
sep_skip=[".",",",";"]
sep_pair={
    "(":")",
    "[":"]",
    "{":"}"
}
sep_level={
    "[":1,
    "(": 2,
    "{":3,
    "]": 1,
    ")": 2,
    "}": 3
}
sep_pair_change={
    ")":"(",
    "]":"[",
    "}":"{"
}
branket_to_seperator={
    "(": ",",
    "[": ",",
    "{": ";"
}

replace_tuples=[("\\u", "uu"),("\\uu", "uu"),("\\r", "rr"),("\\rr","rr"),("\~", "~"),("\\'", "'"),("\\,", ","),
    ("\\\\" ,"rr"),("\ ","r"),("\\u","uu"),("\~","~"),("<extra_id_0>)\'","<extra_id_0>"),("\''","''")]

def sep_pair_completion(tks,seps,mask_id):

    def accept(seps,idx,now_sep,new_pop):
        new_sep=seps[idx]
        while idx < len(seps):
            new_sep = seps[idx]
            if new_sep.value in sep_pair.keys():
                idx=accept(seps,idx+1,new_sep,new_pop)
            elif new_sep.value in sep_skip:
                idx=idx+1
                continue
            elif new_sep.value in sep_pair_change.keys():
                if new_sep.value == sep_pair[now_sep.value]:
                    return idx + 1
                else:
                    if sep_level[now_sep.value] < sep_level[new_sep.value]:
                        new_pop.append(sep_pair[now_sep.value])
                        new_pop.append("MASK")
                        return idx
                    else:
                        new_pop.append(sep_pair_change[new_sep.value])
                        new_pop.append("MASK")
                        idx = idx + 1
                        continue
        new_pop.append(sep_pair[now_sep.value])
        return idx


    new_pop=["MASK"]
    accept_value = accept(seps, 1, seps[0],new_pop)
    mask_tar=tks.pop(mask_id)
    target=mask_id
    last=None
    next=None

    def find_field(tks,target):
        field_sep = None
        for p in range(target, 0, -1):
            mt = tks[p]
            if mt.value in branket_to_seperator.keys():
                return branket_to_seperator[mt.value]

    for i in new_pop:
        last=tks[target-1]
        next=tks[target]

        if i=="MASK":
            insert_ob = tokenizer.Identifier("MASK", position=mask_tar.position)
            # if not isinstance(last,tokenizer.Separator):
            tks.insert(target, insert_ob)
                # tks.insert(target, tokenizer.Separator(find_field(tks,target),position=mask_tar.position))
        else:
            insert_ob = tokenizer.Separator(i,position=mask_tar.position)
            tks.insert(target, insert_ob)


        target=target+1
    return tks


def check_instace(target,type_list):
    for i in type_list:
        if isinstance(target,i):
            return True
    return False

def try_parse_tokens(tks,mask_id,mask_tar):
    try:
        parser = Parser(tks)
        res = parser.parse()
        return res
    except javalang.parser.JavaSyntaxError as e:
        if e.want==javalang.tokenizer.Identifier:
            tks.insert(mask_id+1,tokenizer.Identifier("MASK",mask_tar.position))
            return try_parse_tokens(tks,mask_id+1,mask_tar)
        elif e.want == ".":
            if isinstance(e.at,javalang.tokenizer.Operator):
                tks.insert(mask_id, tokenizer.Separator(";", position=mask_tar.position))
                return try_parse_tokens(tks,mask_id+1,mask_tar)
            elif isinstance(e.at,javalang.tokenizer.Identifier):
                tks.insert(mask_id + 1, tokenizer.Separator(";", position=mask_tar.position))
                tks.insert(mask_id, tokenizer.Separator(";", position=mask_tar.position))
                return try_parse_tokens(tks, mask_id + 1, mask_tar)
            else:
                print("wait")
        elif e.want in tokenizer.Separator.VALUES:
            tks.insert(mask_id + 1, tokenizer.Separator(e.want,position=mask_tar.position))
            return try_parse_tokens(tks, mask_id + 1, mask_tar)
        else:
            print("wait")



def random_parse(s,mask_part,between):

    test='class test{ public void updateLockdownExceptions(ManagedObjectReference _this, String[] users) throws AuthMinimumAdminPermission, RemoteException, RuntimeFault, UserNotFound { Argument[] params = new Argument[2]; params[0 MASK "_this", "ManagedObjectReference", _this); params[1] = new Argument("users", "String[]", users); getWsc().invoke( "UpdateLockdownExceptions", params, null); }}'
    tokens = tokenizer.tokenize(test)
    tks = [i for i in tokens]

    mask_id=-1
    mask_tar=None
    for idx,i in enumerate(tks):
        if i.value == "MASK":
            mask_id=idx
            mask_tar=i
            break
    try:
        seps=[]
        tar_sep=-1
        for i in tks:
            if isinstance(i,tokenizer.Separator):
                seps.append(i)
            if i.value=="MASK":
                tar_sep=len(seps)-1
        tks = sep_pair_completion(tks,seps,mask_id)
        tree = try_parse_tokens(tks,mask_id,mask_tar)
    except Exception as e:
        print("wait")

    tokens = tokenizer.tokenize(s.replace("<extra_id_0>",mask_part))
    tks = [i for i in tokens]
    # token_num=random.randint(3,8)
    # begin_id=random.randint(0,len(tks)-1-token_num)
    # res_tk=-1
    # for i in range(token_num):
    #     m=tks.pop(begin_id)
    #     if i==0:
    #         res_tk=m
    # tks.insert(begin_id,tokenizer.Identifier("MASK",position=res_tk.position))
    # for i in tks:
    #     print(i.value)
    parser = Parser(tks)
    res=parser.parse()




def parse(s,mask_part,between):
    tokens = tokenizer.tokenize(s.replace("<extra_id_0>",mask_part))
    tks = [i for i in tokens]
    field_types=[]
    first_appear=-1
    for idx,i in enumerate(tks):
        pos=i.position.column - 1

        if (not check_instace(i,replace_type)) and (not check_instace(i,not_replace_type)) and (not isinstance(i,tokenizer.Keyword)):
            print("wait")
        if  between[0] <= pos < between[1]:
            if check_instace(i,replace_type):
                i.value="MASK"
            elif isinstance(i,tokenizer.Keyword):
                if i.value == "this" or i.value == "void":
                    tks[idx]=tokenizer.Identifier("MASK",i.position)
                if isinstance(i,tokenizer.BasicType):
                    tks[idx] = tokenizer.Identifier("MASK", i.position)
            field_types.append(type(i).__name__)
            if first_appear==-1:
                first_appear=idx
    if (set(field_types)==set(["Separator"])):
        tks.insert(first_appear,tokenizer.Identifier("MASK",i.position))



    parser = Parser(tks)
    res=parser.parse()
    return res

def normal_parse(text_full,replacement,mask_text="MASK",tuple_id=-1):
    try:
        if tuple_id >0:
            r_tuple=replace_tuples[tuple_id]
            text_rep = text_full.replace(r_tuple[0],r_tuple[1])
        else:
            text_rep = text_full

        tokens = tokenizer.tokenize(text_rep.replace(replacement,mask_text))
        parser = Parser(tokens)
        res=parser.parse()
    except Exception as e:
        if mask_text=="MASK":
            return normal_parse(text_full,replacement,"MASK MASK",tuple_id)
        elif mask_text=="MASK MASK":
            return normal_parse(text_full, replacement, "MASK;MASK;MASK",tuple_id)
        elif tuple_id < len(replace_tuples) - 1:
            return normal_parse(text_full, replacement, "MASK", tuple_id+1)
        else:
            raise e
    return res,text_rep

def getAttr(tree,attr):
    if tree is None:
        print("wait")
    if isinstance(tree, dict):
        return tree[attr]
    else:
        return getattr(tree,attr)

def setAttr(tree,attr,v):
    if isinstance(tree, dict):
        tree[attr]=v
    else:
        setattr(tree,attr,v)

def check_MASK(tree,reses):
    avai = False

    avai_c=getAttr(tree,'avai_childs')

    for t in avai_c:
        if isinstance(t,dict):
            if t['value'] != "MASK":
                avai=True
                break
    if not avai:
        if tree not in reses:
            reses.append(tree)
        return check_MASK(getAttr(tree,'parent'),reses)
    else:
        return reses


def conduct_MASK(res_list):
    delete_types=[]
    delete_tokens=[]
    new_tokens=[]
    new_types=[]
    firstMASK=1
    for i in res_list['tokens']:
        if i["value"]=="MASK":
            delete_tokens.append(i)
            if firstMASK==1:
                firstMASK = i
                new_tokens.append(i)
            delete_types=check_MASK(i['parent'],delete_types)
        else:
            new_tokens.append(i)
    for i in res_list['types']:
        if i not in delete_types:
            new_types.append(i)

    res_list["tokens"]=new_tokens
    res_list["types"]=new_types
    res_list["delete_tokens"]=delete_tokens
    res_list["delete_types"]=delete_types
    if firstMASK==1:
        print("wait")
    # print(res_list["tokens"])
    res_list["MASK_pos"]=firstMASK



def deep_in_tree(tree,elements,between,parent=None,pos=-1,belong=-1,field=None):
    if between is None:
        between=[-1,-1]
    if isinstance(tree, list) or isinstance(tree, set):
        if len(tree) > 0:
            # id=len(elements['types'])
            # if isinstance(parent,dict):
            #     data = {"id": id, "value": "{}_child".format(parent["value"]),
            #             "parent": parent, "avai_childs": []}
            # else:
            #     data={"id":id,"value":"{}_{}".format(type(parent).__name__,parent.attrs[belong]),"parent":parent,"avai_childs":[]}
            # elements['types'].append(data)
            # parent=data
            for idx,i in enumerate(tree):
                child = deep_in_tree(i,elements,between,parent,pos,belong,field=field)
                # if child is not None:
                #     data["avai_childs"].append(child)

        return None
    elif isinstance(tree, str):
        if tree == "":
            return ""
        if between[0] <= pos < between[1]:
            tree="MASK"
        if tree=="MASK":
            elements['field']=field
        id=len(elements['tokens'])+1
        attr = "{}_{}".format(type(parent).__name__, parent.attrs[belong])
        data={"id":-id,"value":tree,"parent":parent,"belong_attr":attr}
        elements['tokens'].append(data)
        return data
    elif isinstance(tree, bool):
        if between[0] <= pos < between[1]:
            tree = "MASK"
        id = len(elements['tokens']) + 1
        attr = "{}_{}".format(type(parent).__name__, parent.attrs[belong])
        data={"id":-id,"value":tree,"parent":parent,"belong_attr":attr}
        elements['tokens'].append(data)
        return data
    elif tree is None:
        return None
    else:
        if isinstance(tree, javalang.tree.This):
            if between[0] <= pos < between[1]:
                value = "MASK"
            else:
                value = "this"
            id_this = len(elements['tokens']) + 1
            attr = "{}_{}".format(type(parent).__name__, parent.attrs[belong])
            data_this = {"id": -id_this, "value": value, "parent": parent,"belong_attr":attr}
            elements['tokens'].append(data_this)

        if parent is not None:
            attr="{}_{}".format(type(parent).__name__,parent.attrs[belong])
            tree.belong_attr=attr

        id = len(elements['types'])

        tree.id=id
        elements['types'].append(tree)
        tree.parent=parent

        # elements['nodes'].append(tree)
        if tree.position is not None:
            pos=tree.position.column
        avai_childs=[]
        if type(tree) in main_type_in:
            field=type(tree).__name__
        for idx,i in enumerate(tree.children):
            if i is not None:
                child=deep_in_tree(i, elements,between,tree, pos,belong=idx,field=field)
                if child is not None:
                    avai_childs.append(child)
            else:
                continue
        tree.avai_childs=avai_childs
        return tree

def find_Parents(tree,childs,id_mapping):
    if (isinstance(tree, dict) and ("parent" not in tree.keys() or tree['parent'] is None)) or (
            hasattr(tree, "parent") and tree.parent is None):
        return childs
    else:
        parent = getAttr(tree, "parent")
        cid = getAttr(tree, "id")
        pid = getAttr(parent, "id")
        childs.append((id_mapping[pid],id_mapping[cid]))
        # childs.append((pid, cid))
    return childs



def build_edges(res_list):
    res = {
        "tokens": [],
        "types": [],
        "edges":{
        "Child": [],
        "NextToken": [],
        "NextUse": []
        }
    }
    def makeID(id):
        if id >= 0:
            return id
        else:
            id=len(res_list['types'])-1+-(id)
            return id
    id_mapping={}
    now_id=0
    edges=[]
    for i in res_list["types"]:
        id_mapping[getAttr(i,"id")]=now_id
        now_id=now_id+1
    for i in res_list["tokens"]:
        id_mapping[getAttr(i,"id")]=now_id
        now_id=now_id+1
    MASK_id=id_mapping[getAttr(res_list["MASK_pos"],"id")]
    for i in res_list['delete_tokens']:
        id_mapping[getAttr(i,"id")]=MASK_id
    for i in res_list['delete_types']:
        id_mapping[getAttr(i,"id")]=MASK_id

    for i in res_list["types"]:
        if isinstance(i,dict):
            res["types"].append((i["value"]))
        else:
            attr=i.belong_attr if hasattr(i,"belong_attr") else "root"
            res["types"].append((type(i).__name__,attr))

    last_token={}
    last_token_id=2
    for i in res_list["tokens"]:
        if i["value"] in last_token.keys():
            res["edges"]['NextUse'].append((id_mapping[last_token[i["value"]]],id_mapping[i['id']]))
        last_token[i["value"]]=i["id"]

        attr = i["belong_attr"]
        res["tokens"].append((i["value"],attr))
        if(last_token_id != 2):
            res["edges"]['NextToken'].append((id_mapping[last_token_id], id_mapping[i['id']]))
        last_token_id=i["id"]
        edges= find_Parents(i, edges,id_mapping)

    for i in res_list["delete_tokens"]:
        edges = find_Parents(i, edges, id_mapping)

    for i in res_list["delete_types"]:
        edges = find_Parents(i, edges, id_mapping)


    res["edges"]["Child"]=edges

    for key,edge_list in res["edges"].items():
        res["edges"][key]=list(set(edge_list))

    res["MASK_id"]=MASK_id


    # min_delete_id=getAttr(res_list['delete_types'][0],"id")
    # min_delete_type=res_list['delete_types'][0]
    # for i in res_list['delete_types']:
    #     if getAttr(i,"id") < min_delete_id:
    #         min_delete_id=getAttr(i,"id")
    #         min_delete_type=i
    #
    # print(type(min_delete_type).__name__)
    # print([type(name).__name__ for name in res_list['delete_types']])



    return res

def get_tokens(split):
    begin_pos=split[0].find("<extra_id_0>")
    end_pos=begin_pos + len(split[1])
    tkss = tokenizer.tokenize(split[0].replace("<extra_id_0>", split[1]))
    tkss =[i for i in tkss]
    full_text=[]
    predict_tar=[]
    first_appear=False
    for i in tkss:
        if begin_pos< i.position.column <= end_pos:
            predict_tar.append(i)
            if not first_appear:
                full_text.append(tokenizer.Identifier("MASK",position=begin_pos))
                first_appear=True
        else:
            full_text.append(i)
    assert first_appear
    full_text=[i.value for i in full_text]
    return (full_text,predict_tar)


def pretreat(file_dir,output_dir="",dict_dir=""):
    ida=0
    input_file=open(file_dir,mode="r",encoding="utf-8")
    text = input_file.readline()
    error_id=[]
    reses=[]
    while text:
        ida = ida+1
        print(ida)
        # if ida <2581:
        #     text = input_file.readline()
        #     continue
        # if ida in [47757]:
        #     text = input_file.readline()
        #     continue
        # if ida > 10000:
        #     break;
        save_text=text
        text=text.replace("\n"," ")        # text=repr(text)
        # text=re.sub(r'\\','l',text)
        split=text.split("	")
        try:
            text_full="class test { " + split[0] + "}"
            tree,text_full=normal_parse(text_full,"<extra_id_0>")
            # begin = text_full.find("<extra_id_0>")
            # end = begin + len(split[1])


            res_list= {
                "tokens":[],
                "types":[]
            }
            tks=deep_in_tree(tree.children[2][0],res_list,None)
            if "field" not in res_list.keys() or res_list['field'] is None:
                print("wait")
            conduct_MASK(res_list)
            res=build_edges(res_list)
            res["text"]=get_tokens([text_full,split[1]])

            res['field']=res_list['field']
        except Exception as e:
            error_id.append((ida,e))
            print("error")
            text=input_file.readline()
            continue


        reses.append(res)
        text = input_file.readline()
    print("end")
    reses_dict={"reses":reses,"errors":error_id}
    cache_file=open(res_dir,mode="wb")
    pickle.dump(reses_dict,cache_file)




# text="class test{public void tuneEnd(Tune tune, AbcNode abcRoot) { a.b().c();} } "
# tree=javalang.parse.parse(text)


parser = argparse.ArgumentParser(description='Jenkins pipline parameters')

parser.add_argument('--file_name', type=str, default='train_java_construct', help='uuid value')
args = parser.parse_args()

file_name = args.file_name

file_dir="/root/CCSG/data/{}.tsv".format(file_name)
res_dir="/root/CCSG/data/data_cache/{}.pkl".format(file_name)

pretreat(file_dir)
