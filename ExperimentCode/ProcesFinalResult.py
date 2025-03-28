# -*- coding: utf-8 -*-
import torch
import shutil
import subprocess
import openai
import pandas as pd
import os
import re
import json
import time
import tiktoken
from tqdm import tqdm
import traceback
import glob
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from Deal import Compile_Test_INFO
from Deal import FeedbackPrompt


# current_dir = os.path.dirname(__file__) #./PipLine
current_dir = os.path.dirname(os.path.abspath(__file__))    # os.path.abspath：获取本脚本的绝对路径
AgentTesterDir = os.path.dirname(current_dir)                # os.path.dirname：获取父目录名

testedRepo_PATH = os.path.join(AgentTesterDir, "Repos")
# model_path = "glm-4"
model_path = "glm-zero-preview"
# model_path = "deepseek-r1:8b"

class ProceFinalResult:
    def __init__(self, repo_name):
        Json_file_Path = os.path.join(AgentTesterDir, "RepoData", repo_name)
        self.repo_name = repo_name.replace(".json", "")
        # pyPath in contain_intention. The result in the folder is from the InitialPhrase_Experiment.
        if "CodeLlama-34b-Instruct" in model_path:
            self.sub_save_dir = 'CodeLlama'  # CodeLlama; WizardCoder
        elif "CodeFuse-CodeLlama" in model_path:
            self.sub_save_dir = "CodeFuse"
        elif "glm-zero-preview" in model_path:
            self.sub_save_dir = os.path.basename(Json_file_Path).replace(".json", "")

        first_dir = "Iterate"
        self.C_GeneratedTest_Path = os.path.join(current_dir, first_dir, self.sub_save_dir, 'GeneratedTest')
        self.C_Surefire_reports_Path = os.path.join(current_dir, first_dir, self.sub_save_dir, 'Surefire_reports')
        self.C_LogINFO_Path = os.path.join(current_dir, first_dir, self.sub_save_dir, 'LogINFO')
        self.pred_1 = os.path.join(current_dir, first_dir, self.sub_save_dir, 'final_result.json')

        # Path in iterate. 基于上面的文件夹，再进一步进行推理，得到迭代之后的结果.
        dir_Name = "IterateResultDeal"
        self.GeneratedTest_PATH = os.path.join(current_dir, dir_Name, self.sub_save_dir, 'GeneratedTest')
        self.Final_result = os.path.join(current_dir, dir_Name, self.sub_save_dir, 'final_result.json')

        self.boolean(self.GeneratedTest_PATH)

        self.LoadFile()

    def boolean(self, file_path):
        if not os.path.exists(file_path):
            print('Creat folder....')
            os.makedirs(file_path)
        else:
            shutil.rmtree(file_path)
            os.makedirs(file_path)

    def LoadFile(self):
        outputList = []
        self.count = 0

        with open(self.pred_1, 'r', encoding='utf-8') as f:
            file_con = json.load(f)  # 解析整个JSON文件内容
            for con in tqdm(file_con):
                ori_test_Path = con['original_path']
                generated_path_old = con['generated_path']
                FocalMethodInfo = os.path.basename(generated_path_old)
                shutil.copy2(generated_path_old, self.GeneratedTest_PATH)
                generated_path = os.path.join(self.GeneratedTest_PATH, os.path.basename(generated_path_old))
                Compile_result = con['Compile_result']
                Test_result = con['Test_result']

                with open(generated_path, 'r', encoding='utf-8') as f:
                    FixGencont = f.read()

                if Compile_result == 0: continue
                elif Test_result == 1:
                    finalCont = {"original_path": ori_test_Path,
                                 "generated_path": generated_path,
                                 "IterateTimes": 0,
                                 "Compile_result": Compile_result,
                                 "Test_result": Test_result}
                    outputList.append(finalCont)
                    continue

                self.DriveTest_Info(FocalMethodInfo)
                project_name = os.path.basename(Json_file_Path).replace(".json", "")
                GenJava = os.path.basename(generated_path)
                # 找编译日志和Surefire测试报告文件：
                compile_logInfo_path = [file for file in glob.glob(self.C_LogINFO_Path + '/*') if os.path.basename(file) == GenJava][0]
                Surefire_reports_dst_file = [file for file in glob.glob(self.C_Surefire_reports_Path + '/*') if
                                                 GenJava.replace(".java", '.xml') in os.path.basename(file)]
                if len(Surefire_reports_dst_file) == 0: Surefire_reports_dst_file = ['None']
                # 确定修复标签：
                if Compile_result == 0: repairTag = "compileRepair"
                elif Compile_result == 1 and Test_result == 0: repairTag = "testRepair"
                Composit_prompt, proc_test_list_INFO, findClassInfo = self.Collect_Info(compile_logInfo_path,
                                                                    Surefire_reports_dst_file[0],
                                                                    generated_path,
                                                                    ori_test_Path, False, findClassInfo)

                # 如果复合提示中包含生成断言语句，则提取测试方法并生成新的测试代码。
                # 然后将新的测试代码写入生成的Java文件中。最后，将测试结果添加到outputList中。
                if "[Generate an assertion statement here]" in Composit_prompt:
                    Test_Method = Composit_prompt.split("# Test Method", 1)[-1].split("# Instruction", 1)[0].split("[Generate an assertion statement here]",1)[0] + "\n}"
                    Test_Code = FixGencont.split("ESTest_scaffolding {", 1)[0] + "ESTest_scaffolding {\n" + Test_Method + "\n}"
                    with open(generated_path, 'w', encoding='utf-8') as f:
                        f.write(Test_Code)

                    finalCont = {"original_path": ori_test_Path,
                                 "generated_path": generated_path,
                                 "IterateTimes": 0,
                                 "Compile_result": Compile_result,
                                 "Test_result": 1}
                    outputList.append(finalCont)
                else: continue

        with open(self.Final_result, "w", encoding="utf-8") as f:
            json.dump(outputList, f, indent=2)

    # 处理和生成测试代码的模板，从一个JSON文件中读取测试信息，并根据这些信息生成测试代码的框架和示例
    def DriveTest_Info(self, FocalMethodInfo):
        with open(Json_file_Path, 'r', encoding='utf-8') as f:
            data_pair = json.load(f)
        # 查找测试信息:
        ori_test_Path = [data["Test_method"]["TestInfo"] for data in data_pair if len(data['Under_test_method']) and data["Under_test_method"]["Method_statement"] == FocalMethodInfo.split("#")[-1].replace(".java","") and FocalMethodInfo.split("#")[0] in data['Test_method']['TestInfo']][0]

        # 生成测试框架的路径和代码：
        TestScaffoldPath = ori_test_Path.split("###")[0].replace(".java", "_scaffolding.java")  # 测试框架的文件路径
        ScaffoldingCode = [data['Test_method']['scaffoldingCode'] for data in data_pair if
                           data["Test_method"]["TestInfo"] == ori_test_Path][0]     # 从JSON文件中读取的测试框架代码
        self.boolean(os.path.dirname(TestScaffoldPath))
        with open(TestScaffoldPath, 'w', encoding='utf-8') as f:    # 将测试框架代码写入文件
            f.write(ScaffoldingCode)
        self.testCodeShell = [data['Test_method']['TestCodeShell'] for data in data_pair if
                              data["Test_method"]["TestInfo"] == ori_test_Path][0]

        self.Under_test_method_INFO = [data["Under_test_method"] for data in data_pair if data["Test_method"]["TestInfo"] == ori_test_Path][0]
        self.Junit_version = self.Under_test_method_INFO['Junit_version']

        Focal_class = self.Under_test_method_INFO['Class_declaration']
        Filed = self.Under_test_method_INFO['Filed'] + "\n"
        constructors = self.Under_test_method_INFO['constructors'] + "\n"
        self.Focal_Method_Info = self.Under_test_method_INFO["Method_body"]
        PL_Focal_Method = Focal_class + '\n' + Filed + constructors + '\n' + '# Focal method\n' + self.Focal_Method_Info + "\n}"
        self.PL_Focal_Method = '\n'.join(filter(lambda x: x.strip(), PL_Focal_Method.split('\n')))
        self.focal_method_name = self.Under_test_method_INFO['Method_statement']
        self.TestMethodExample = [data['Test_method']['TestMethodBody'] for data in data_pair if
                                  data["Test_method"]["TestInfo"] == ori_test_Path][0]

    # 收集和处理测试信息
    def Collect_Info(self, compile_logInfo_path, Surefire_reports_dst_file, gen_test_PATH,
                     ori_test_Path, re_generate_Tag, findClassInfo):

        test_instance = Compile_Test_INFO.TestINFO(Surefire_reports_dst_file, compile_logInfo_path)
        proc_test_list_INFO = test_instance.TetsINFO_deal()
        Method_intention = ""
        class_instance = FeedbackPrompt.TestPrompt(proc_test_list_INFO[0], gen_test_PATH, Method_intention,
                                                   self.Focal_Method_Info, self.focal_method_name)
        Composit_prompt = class_instance.Test_deal()

        return Composit_prompt, proc_test_list_INFO, findClassInfo


if __name__ == "__main__":
    # projects_name = ['sachin-handiekar_jInstagram.json', 'tabulapdf_tabula-java.json', 'Zappos_zappos-json.json']
    projects_name = ['sachin-handiekar_jInstagram.json']

    for project_name in projects_name:
        Json_file_Path = os.path.join(AgentTesterDir, "RepoData", project_name)
        # ProceFinalResult(project_name.replace(".json", ""))
        ProceFinalResult(project_name)
