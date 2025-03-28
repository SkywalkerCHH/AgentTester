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
from ProcesFinalResult import ProceFinalResult

# TODO modify this path
# java_home = "/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home"
java_home = "C:/Program Files/Java/jdk1.8.0_152"
os.environ["JAVA_HOME"] = java_home
env = os.environ.copy()
env['JAVA_TOOL_OPTIONS'] = '-Duser.language=en -Duser.country=US'
current_dir = os.path.dirname(os.path.abspath(__file__))
AgentTesterDir = os.path.dirname(current_dir)    # 获取当前脚本所在目录的父目录

testedRepo_PATH = os.path.join(AgentTesterDir, "Repos")  # 定义测试仓库的路径

# model_path = "glm-4"
model_path = "glm-zero-preview"
# model_path = "deepseek-r1:8b"


class ChatGptTester:
    def __init__(self, repo_name):

        # Path in root
        self.Result_PATH = os.path.join(AgentTesterDir, "RepoData")  # 结果数据路径 Data pair path
        self.testedRepo_PATH = os.path.join(AgentTesterDir, "Repos")  # 存储已测试仓库路径 repo path
        self.repo_name = repo_name
        # Path in contain_intention. The result in the folder is from the InitialPhrase_Experiment.py
        if "CodeLlama-34b-Instruct" in model_path:
            self.sub_save_dir = 'CodeLlama'  # CodeLlama; WizardCoder
        elif "CodeFuse-CodeLlama" in model_path:
            self.sub_save_dir = "CodeFuse"
        elif "glm-zero-preview" in model_path:
            self.sub_save_dir = os.path.basename(Json_file_Path).replace(".json", "")
            openai.api_base = ""
            # TODO SET API_KEY
            openai.api_key = ""
            
        self.C_GeneratedTest_Path = os.path.join(current_dir, 'Contain_intention', self.sub_save_dir, 'GeneratedTest_2')
        self.C_LogINFO_Path = os.path.join(current_dir, 'Contain_intention', self.sub_save_dir, 'LogINFO')
        self.C_Surefire_reports_Path = os.path.join(current_dir, 'Contain_intention', self.sub_save_dir, 'Surefire_reports')
        self.pred_1 = os.path.join(current_dir, "Contain_intention", self.sub_save_dir, 'result_2.json')
        # Path in iterate. 基于上面的文件夹，再进一步进行推理，得到迭代之后的结果.
        dir_Name = "Iterate"
        self.original_java_PATH = os.path.join(current_dir, dir_Name, self.sub_save_dir, 'original_java')
        self.LogINFO_PATH = os.path.join(current_dir, dir_Name, self.sub_save_dir, 'LogINFO')
        self.Surefire_reports_dest_path = os.path.join(current_dir, dir_Name, self.sub_save_dir, 'Surefire_reports')
        self.GeneratedTest_PATH = os.path.join(current_dir, dir_Name, self.sub_save_dir, 'GeneratedTest')
        self.RepairProcess = os.path.join(current_dir, dir_Name, self.sub_save_dir, 'RepairProcess')
        self.Final_result = os.path.join(current_dir, dir_Name, self.sub_save_dir, 'final_result.json')
    
        self.repairCompile_result = os.path.join(current_dir, dir_Name, self.sub_save_dir, 'RepairCompile.json')
        self.repairTest_result = os.path.join(current_dir, dir_Name, self.sub_save_dir, 'RepairTest.json')

        self.boolean(self.original_java_PATH)
        self.boolean(self.LogINFO_PATH)
        self.boolean(self.Surefire_reports_dest_path)
        self.boolean(self.GeneratedTest_PATH)
        self.boolean(self.RepairProcess)

        self.unit_instance = Unit(model_path)  # class instance

        self.LoadFile()

    def boolean(self, file_path):
        if not os.path.exists(file_path):
            print('Creat folder....')
            os.makedirs(file_path)  # 创建文件夹
        else:
            shutil.rmtree(file_path)    # 如果路径存在，则删除路径及其所有内容
            os.makedirs(file_path)      # 重新创建

    # 读取一个JSON文件，解析其中的内容，并根据解析结果执行一系列操作，包括复制文件、执行Git操作、收集信息、修复代码等，将结果保存到新的JSON文件中：
    def LoadFile(self):
        self.count = 0
        with open(self.pred_1, 'r', encoding='utf-8') as f:
            file_con = json.load(f)  # 解析整个JSON文件内容
            for con in tqdm(file_con):
                findClassInfo = ""  # store collected class Info
                ori_test_Path = con['original_path']    # 从con中提取键值
                generated_path_old = con['generated_path']
                FocalMethodInfo = os.path.basename(generated_path_old)  # basename：获取文件名
                shutil.copy2(generated_path_old, self.GeneratedTest_PATH)   # 将生成的文件复制到指定目录
                generated_path = os.path.join(self.GeneratedTest_PATH, os.path.basename(generated_path_old))
                Compile_result = con['Compile']
                Test_result = con['Test']

                # 如果 在 initial phrase 已经正确，则不用修复
                if Compile_result == 1 and Test_result == 1:
                    finalCont = {"original_path": ori_test_Path,
                                 "generated_path": generated_path,
                                 "IterateTimes": 0,
                                 "Compile_result": Compile_result,
                                 "Test_result": Test_result
                                 }
                    # 覆写原Iterate/./final_result.json文件
                    try:
                        with open(self.Final_result, "r", encoding="utf-8") as f:
                            final_result = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        final_result = []
                        # 添加新的数据
                    final_result.append(finalCont)

                    with open(self.Final_result, "w", encoding="utf-8") as f:
                        json.dump(final_result, f, indent=4)  # 记录结果到JSON文件
                    continue

                project_name = os.path.basename(Json_file_Path).replace(".json", "")

                try:
                    # 从 source json file 当中检索信息
                    self.DriveTest_Info(FocalMethodInfo)

                    GenJava = os.path.basename(generated_path)
                    with open(generated_path, 'r', encoding='utf-8') as f:
                        FixGencont = f.read()  # first repair code

                    compile_logInfo_path = [file for file in glob.glob(self.C_LogINFO_Path + '/*') if os.path.basename(file) == GenJava][0]
                    Surefire_reports_dst_file = [file for file in glob.glob(self.C_Surefire_reports_Path + '/*') if GenJava.replace(".java",'.xml') in os.path.basename(file)]
                    if len(Surefire_reports_dst_file) == 0: Surefire_reports_dst_file=['None']

                    if Compile_result == 0: repairTag = "compileRepair"
                    elif Compile_result == 1 and Test_result == 0: repairTag = "testRepair"

                    # File ID
                    self.count = self.count + 1

                    print(f"Deal: {self.count}; repairTag: {repairTag}; ", ori_test_Path)

                    Composit_prompt, proc_compile_list_INFO, proc_test_list_INFO, findClassInfo = self.Collect_Info(Compile_result,
                                                                                                     Test_result,
                                                                                                     compile_logInfo_path,
                                                                                                     Surefire_reports_dst_file[0],
                                                                                                     generated_path,
                                                                                                     ori_test_Path, False, findClassInfo)

                    if (len(proc_compile_list_INFO) == 0 and len(proc_test_list_INFO) == 0) or len(Composit_prompt) == 0:
                        finalCont = {"original_path": ori_test_Path,
                                     "generated_path": generated_path,
                                     "IterateTimes": 0,
                                     "Compile_result": Compile_result,
                                     "Test_result": Test_result
                                     }
                        # 读取现有的JSON文件内容，如果文件存在且不是空的
                        try:
                            with open(self.Final_result, "r", encoding="utf-8") as f:
                                final_result = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            final_result = []
                        # 添加新的数据
                        final_result.append(finalCont)

                        with open(self.Final_result, "w", encoding="utf-8") as f:
                            json.dump(final_result, f, indent=4)  # 记录结果到JSON文件
                        continue

                    self.IteratePred(repairTag, generated_path, FixGencont, Composit_prompt, ori_test_Path, os.path.basename(generated_path).replace(".java", ""), findClassInfo)

                except Exception as e:
                    traceback.print_exc()
                finally:
                    test_path = os.path.join(testedRepo_PATH, project_name, "src", "test")
                    self.boolean(test_path)
                    print("Project reset.")

    # 读取 JSON 文件中的测试信息，并根据给定的 FocalMethodInfo（焦点方法信息）提取相关信息，
    # 包括测试脚手架代码(Scaffold)、测试代码壳、被测方法信息、JUnit 版本、焦点类信息、字段、构造函数和焦点方法信息。
    def DriveTest_Info(self, FocalMethodInfo):
        with open(Json_file_Path, 'r', encoding='utf-8') as f:
            data_pair = json.load(f)

        ori_test_Path = [data["Test_method"]["TestInfo"] for data in data_pair if len(data['Under_test_method']) and data["Under_test_method"]["Method_statement"] == FocalMethodInfo.split("#")[-1].replace(".java","") and FocalMethodInfo.split("#")[0] in data['Test_method']['TestInfo']][0]

        TestScaffoldPath = ori_test_Path.split("###")[0].replace(".java", "_scaffolding.java")
        ScaffoldingCode = [data['Test_method']['scaffoldingCode'] for data in data_pair if data["Test_method"]["TestInfo"] == ori_test_Path][0]

        self.boolean(os.path.dirname(TestScaffoldPath))
        with open(TestScaffoldPath, 'w', encoding='utf-8') as f:
            f.write(ScaffoldingCode)

        self.testCodeShell = [data['Test_method']['TestCodeShell'] for data in data_pair if data["Test_method"]["TestInfo"] == ori_test_Path][0]
        self.Under_test_method_INFO = [data["Under_test_method"] for data in data_pair if data["Test_method"]["TestInfo"] == ori_test_Path][0]
        self.Junit_version = self.Under_test_method_INFO['Junit_version']

        Focal_class = self.Under_test_method_INFO['Class_declaration']
        Filed = self.unit_instance.commentDelete(self.Under_test_method_INFO['Filed']) + "\n"
        constructors = self.unit_instance.commentDelete(self.Under_test_method_INFO['constructors']) + "\n"
        self.Focal_Method_Info = self.unit_instance.commentDelete(self.Under_test_method_INFO["Method_body"])
        PL_Focal_Method = Focal_class + '\n' + Filed + constructors + '\n' + '# Focal method\n' + self.Focal_Method_Info + "\n}"
        self.PL_Focal_Method = '\n'.join(filter(lambda x: x.strip(), PL_Focal_Method.split('\n')))
        self.focal_method_name = self.Under_test_method_INFO['Method_statement']
        self.TestMethodExample = [data['Test_method']['TestMethodBody'] for data in data_pair if data["Test_method"]["TestInfo"] == ori_test_Path][0]

    # 迭代地进行代码修复，并根据编译和测试结果更新提示信息。
    # 它首先尝试修复编译错误，然后修复测试错误，直到代码能够成功编译和通过测试，或者达到指定的迭代次数上限。
    def IteratePred(self, repairTag, generated_path, FixGencont, Composit_prompt, ori_test_Path, fixedClassName, findClassInfo):
        re_generate_Tag = False
        project_name = os.path.basename(Json_file_Path).replace(".json", "")
        TAG = "TEST"
        error_numbers = []
        TotalIter = 0  # 总的迭代次数
        iter = 0  # compile 和 Test的修复次数
        IterCompile, IterTest = 1, 0
        while True:
            print(f'----------------{ori_test_Path}----------------')
            print(Composit_prompt)
            # Out_Txtdir 路径为 '...\\ExperimentCode\\Iterate\\...\\RepairProcess\\..._prompt.txt'
            Out_Txtdir = os.path.join(self.RepairProcess,
                                      os.path.basename(ori_test_Path.split("###")[0]) + "_" +
                                      fixedClassName.split("#")[1] + "_prompt.txt")
            with open(Out_Txtdir, 'a', encoding='utf-8') as f:
                f.write(f"{TotalIter}-->{IterCompile | IterTest}-->{iter}-->{repairTag}\n" + Composit_prompt + "\n\n########\n\n")
            print(f'-----------------------------------------------')

            TotalIter = TotalIter + 1
            pattern = re.compile(r'//\s*original\s+test\s+path:\s*[\S\s]*?\n')
            Composit_prompt = pattern.sub('', Composit_prompt)

            Gen_test_method, import_statement = self.unit_instance.method_pred_unit(Composit_prompt, True)
            TestFilePath = ori_test_Path.split("###")[0]
            focal_method_name = fixedClassName.split("#")[1]
            Dtest_para = self.file_write(generated_path, Gen_test_method, TestFilePath, self.testCodeShell,
                                                           import_statement+"\nimport java.util.*;\nimport java.lang.*;\n", focal_method_name)

            compile_result, test_result, compile_logInfo_path, Surefire_reports_dst_file = self.adhoc_excute(Dtest_para,
                                                                                                   generated_path,
                                                                                                   TestFilePath,
                                                                                                   testedRepo_PATH,
                                                                                                   project_name,
                                                                                                   "4")
            # 获得 error number
            with open(compile_logInfo_path, 'r', encoding='utf-8') as f:
                copile_info = f.read()
            number_match = re.search(r'\d+ error', copile_info)     # 检查编译日志文件中是否包含错误信息
            errors = 0
            if number_match:
                errors = int(number_match.group(0).replace(" error", ""))

            # 如果修之前是 compile=1 and test=0，修复后是 compile_result=0, 则将原来的 test method 保存
            if repairTag == "testRepair" and compile_result == 0:
                with open(generated_path, 'w', encoding='utf-8') as f:
                    f.write(FixGencont)
                # gen_test_PATH, gen_test_cont, package_name = self.file_write(FixGencont, ori_test_Path,fixedClassName)
                IterTest = IterTest + 1

            # 修复编译错误，并保存结果
            if repairTag == "compileRepair":
                outDict = {"count": self.count,
                           "original_path": ori_test_Path,
                           "generated_path": generated_path,
                           "Compile": f"TotalIter:{TotalIter}-->IterCompile:{IterCompile}-->Non_iter:{iter}",
                           "repairTag": repairTag,
                           "Compile_result": compile_result,
                           "Test_result": test_result,
                           "CompileError_number": errors}

                try:
                    with open(self.repairCompile_result, "r", encoding="utf-8") as f:
                        repaircompile_result = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    repaircompile_result = []
                repaircompile_result.append(outDict)

                with open(self.repairCompile_result, "w", encoding="utf-8") as f:
                    json.dump(repaircompile_result, f, indent=4)  # 记录结果到JSON文件

                if compile_result == 1 and test_result == 1: 
                    finalCont = {"original_path": ori_test_Path,
                                 "generated_path": generated_path,
                                 "IterateTimes": TotalIter,
                                 "Compile_result": compile_result,
                                 "Test_result": test_result}

                    try:
                        with open(self.Final_result, "r", encoding="utf-8") as f:
                            final_result = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        final_result = []
                    final_result.append(finalCont)

                    with open(self.Final_result, "w", encoding="utf-8") as f:
                        json.dump(final_result, f, indent=4)  # 记录结果到JSON文件
                    break

                if compile_result == 0:
                    IterCompile = IterCompile + 1
                    error_numbers.append(errors)
                    # # 如果上一次修改之后的错误总数和这一次修改的编译错误总数，没有发生变化，则为一次无效的修改。继续修复
                    if len(error_numbers) >= 2 and error_numbers[-1] >= error_numbers[-2]:
                        iter = iter + 1
                        re_generate_Tag = True

                elif compile_result == 1 and test_result == 0:  # 转到 test repair 步骤
                    TAG = "COMPILE>TEST"
                    repairTag = "testRepair"

            # 修复测试错误，并保存结果
            if repairTag == "testRepair":
                outDict = {"count": self.count,
                           "original_path": ori_test_Path,
                           "generated_path": generated_path,
                           "Test": f"TotalIter:{TotalIter}-->IterCompile:{IterTest}-->TAG:{TAG}",
                           "repairTag": repairTag,
                           "Compile_result": compile_result,
                           "Test_result": test_result}

                try:
                    with open(self.repairTest_result, "r", encoding="utf-8") as f:
                        repairtest_result = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    repairtest_result = []
                repairtest_result.append(outDict)

                with open(self.repairTest_result, "w", encoding="utf-8") as f:
                    json.dump(repairtest_result, f, indent=4)  # 记录结果到JSON文件

                if compile_result == 1 and test_result == 1:
                    finalCont = {"original_path": ori_test_Path,
                                 "generated_path": generated_path,
                                 "IterateTimes": TotalIter,
                                 "Compile_result": compile_result,
                                 "Test_result": test_result}

                    try:
                        with open(self.Final_result, "r", encoding="utf-8") as f:
                            final_result = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        final_result = []
                    final_result.append(finalCont)

                    with open(self.Final_result, "w", encoding="utf-8") as f:
                        json.dump(final_result, f, indent=4)  # 记录结果到JSON文件
                    break

                elif compile_result == 1 and test_result == 0:
                    IterTest = IterTest + 1

            # iter为无效修复次数，IterTest为修复测试次数
            if iter > 5 or IterTest > 5 or TotalIter > 10:
                finalCont = {"original_path": ori_test_Path,
                             "generated_path": generated_path,
                             "IterateTimes": TotalIter,
                             "Compile_result": compile_result,
                             "Test_result": test_result}

                try:
                    with open(self.Final_result, "r", encoding="utf-8") as f:
                        final_result = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    final_result = []
                final_result.append(finalCont)

                with open(self.Final_result, "w", encoding="utf-8") as f:
                    json.dump(final_result, f, indent=4)  # 记录结果到JSON文件
                break

            # obtain the prompt for next prediction
            Composit_prompt, proc_compile_list_INFO, proc_test_list_INFO, findClassInfo = self.Collect_Info(compile_result,
                                                                                             test_result,
                                                                                             compile_logInfo_path,
                                                                                             Surefire_reports_dst_file,
                                                                                             generated_path,
                                                                                             ori_test_Path, re_generate_Tag, findClassInfo)

    # 将测试方法插入到生成的Java测试类文件中，并确保必要的导入包被正确添加。
    def file_write(self, generated_path, test_method, TestFilePath, TestCodeShell,  Test_Import_info, focal_method_name):

        package_name = [code for code in TestCodeShell.split("\n") if "package " in code and ";" in code][0].replace("package ", "").replace(";", "").strip()
        class_name = os.path.basename(TestFilePath).replace(".java", "")

        # generated_path 当中所有的import取出来，然后去重
        with open(generated_path, 'r', encoding='utf-8') as f:
            gen_cont = f.read()
        Test_Import_infoList = Test_Import_info.split("\n")
        TestImportInfo = []
        for Imp in Test_Import_infoList:
            if Imp in gen_cont or "org.junit" in Imp: continue
            TestImportInfo.append(Imp)

        gen_cont = gen_cont.replace("\nimport ", "\n"+"\n".join(TestImportInfo)+"\nimport ", 1)
        new_gen_cont = gen_cont.split("@Test\n")[0] + "\n" + test_method + "\n}\n"
        with open(generated_path, 'w', encoding='utf-8') as f:
            f.write(new_gen_cont)

        with open(TestFilePath, 'w', encoding='utf-8') as f:
            f.write(new_gen_cont)

        return package_name+"."+class_name

    # 执行test 和 compile
    def adhoc_excute(self, Dtest_para, Gen_TestfilePath, TestFilePath, testedRepo_PATH, project_name, JUNIT_VERSION):

        excute_path = os.path.join(testedRepo_PATH, project_name)
        os.chdir(excute_path)

        mvn_compile = ['mvn', 'test-compile', '-Dcheckstyle.skip=true']
        mvn_test = ['mvn', 'test', '-Dcheckstyle.skip=true']
        if JUNIT_VERSION == 5:
            mvn_compile = ['mvn', 'test-compile', '-Dtest.engine=junit-jupiter', '-Dcheckstyle.skip=true']
            mvn_test = ['mvn', 'test', '-Dtest.engine=junit-jupiter', '-Dcheckstyle.skip=true']
        write_cont, compile_result, test_result = self.Compile_Test_sub_unit(mvn_compile, mvn_test, TestFilePath)

        if compile_result != 1 and "[ERROR] COMPILATION ERROR :" not in write_cont and "Could not resolve " in write_cont:
                mvn_install = [ 'mvn', 'clean', 'install']
                mvn_result = subprocess.run(mvn_install, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                             universal_newlines=True, shell=True)
                if "BUILD SUCCESS" in mvn_result.stdout or "BUILD SUCCESS" in mvn_result.stderr:
                    write_cont, compile_result, test_result = self.Compile_Test_sub_unit(mvn_compile, mvn_test, TestFilePath)
        os.chdir(current_dir)

        if compile_result == 0 and "[ERROR] COMPILATION ERROR :" not in write_cont: raise Exception("Mvn execute failed")
        compile_logInfo_path = os.path.join(self.LogINFO_PATH, os.path.basename(Gen_TestfilePath))
        with open(compile_logInfo_path, 'w', encoding='utf-8') as f:
            f.write(write_cont)
        Surefire_reports_dst_file = self.Surefire_reports_TEST_info(write_cont, os.path.basename(Gen_TestfilePath), Dtest_para)

        return compile_result, test_result, compile_logInfo_path, Surefire_reports_dst_file

    # 收集和处理编译和测试过程中的信息，并生成一个包含错误信息和提示信息的复合提示：
    def Collect_Info(self, compile_result, test_result, compile_logInfo_path, Surefire_reports_dst_file, gen_test_PATH, ori_test_Path, re_generate_Tag, findClassInfo):
        proc_compile_list_INFO, proc_test_list_INFO, Composit_prompt = "", "", ""
        if compile_result == 0:
            # 处理编译的错误信息：Out_dict = {"ERROR_MESSAGE": str, "Class_Name": str, "ERROR_LINE": str}
            compile_instance = Compile_Test_INFO.CompileInfo(compile_logInfo_path, self.sub_save_dir, gen_test_PATH)
            proc_compile_list_INFO = compile_instance.Call_errorDeal()  # Call_errorDeal函数处理编译错误信息，并存储结果
            # 重新生成意图
            if re_generate_Tag: Method_intention = self.unit_instance.intention_unit(self.PL_Focal_Method, self.focal_method_name)
            else:Method_intention = ""
            class_instance = FeedbackPrompt.CompilePrompt(proc_compile_list_INFO[0], gen_test_PATH,
                                                          ori_test_Path.split("###")[0].replace("/test/", '/main/').replace("_ESTest.java", ".java"), re_generate_Tag, Method_intention, self.PL_Focal_Method, self.repo_name, findClassInfo)
            Composit_prompt, findClassInfo = class_instance.Compile_deal()

        if compile_result == 1 and test_result == 0:
            # 处理test运行的错误信息:
            # TEST_INFO_dict = {"FILE_NAME":os.path.basename(xml_file_path), "ERROR_MESSAGE":str, "ERROR_LINE":str}
            test_instance = Compile_Test_INFO.TestINFO(Surefire_reports_dst_file, compile_logInfo_path)
            proc_test_list_INFO = test_instance.TetsINFO_deal()
            Method_intention = self.unit_instance.intention_unit(self.PL_Focal_Method, self.focal_method_name)
            # Method_intention = ""
            class_instance = FeedbackPrompt.TestPrompt(proc_test_list_INFO[0], gen_test_PATH, Method_intention, self.Focal_Method_Info, self.focal_method_name)
            Composit_prompt = class_instance.Test_deal()

        if len(proc_compile_list_INFO) == 0 and len(proc_test_list_INFO) == 0:
            print("No information in proc_compile_list_INFO and proc_test_list_INFO")

        return Composit_prompt, proc_compile_list_INFO, proc_test_list_INFO, findClassInfo

    # 将生成的测试代码中的类名替换为原始类名，并将修改后的代码保存到指定的文件路径中
    def Test_to_project(self, gen_class_name, ori_class_name, gen_test_cont, ori_PATH):
        pattern = re.compile(f"(?<![a-zA-Z]){gen_class_name}|{gen_class_name}(?![a-zA-Z])")   # 表示匹配的类名前后不能是字母
        changed_code = pattern.sub(ori_class_name, gen_test_cont)
        with open(ori_PATH, 'w', encoding='utf-8') as f:
            f.write(changed_code)

    # 编译和测试一个Maven项目中的单元测试，并返回编译和测试的结果
    def Compile_Test_unit(self, pro_name, sub_project_name, test_file_name, test_path, Dtest_para, JUNIT_VERSION):
        excute_path = os.path.join(self.testedRepo_PATH, pro_name)
        os.chdir(excute_path)
        if sub_project_name != "":
            mvn_compile = ['mvn', '-pl', sub_project_name, f'-Dtest={Dtest_para}', 'test-compile',
                           '-Dcheckstyle.skip=true']
            mvn_test = ['mvn', '-pl', sub_project_name, f'-Dtest={Dtest_para}', 'test',
                        '-Dcheckstyle.skip=true']
            if JUNIT_VERSION == 5:
                mvn_compile = ['mvn', '-pl', sub_project_name, f'-Dtest={Dtest_para}', 'test-compile',
                               '-Dtest.engine=junit-jupiter', '-Dcheckstyle.skip=true']
                mvn_test = ['mvn', '-pl', sub_project_name, f'-Dtest={Dtest_para}', 'test',
                            '-Dtest.engine=junit-jupiter', '-Dcheckstyle.skip=true']

        else:
            mvn_compile = ['mvn', f'-Dtest={Dtest_para}', 'test-compile', '-Dcheckstyle.skip=true']
            mvn_test = ['mvn', f'-Dtest={Dtest_para}', 'test', '-Dcheckstyle.skip=true']
            if JUNIT_VERSION == 5:
                mvn_compile = ['mvn', f'-Dtest={Dtest_para}', 'test-compile',
                               '-Dtest.engine=junit-jupiter', '-Dcheckstyle.skip=true']
                mvn_test = ['mvn', f'-Dtest={Dtest_para}', 'test', '-Dtest.engine=junit-jupiter',
                            '-Dcheckstyle.skip=true']

        write_cont, compile_result, test_result = self.Compile_Test_sub_unit(mvn_compile, mvn_test, test_path)

        # 未能正确的执行mvn 指令。此时首先需要执行 mvn clean
        if compile_result != 1 and "[ERROR] COMPILATION ERROR :" not in write_cont and "Could not resolve dependenci" in write_cont:
            mvn_install = ['mvn', 'clean', 'install']
            mvn_result = subprocess.run(mvn_install, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True, shell=True)
            if "BUILD SUCCESS" in mvn_result.stdout or "BUILD SUCCESS" in mvn_result.stderr:
                write_cont, compile_result, test_result = self.Compile_Test_sub_unit(mvn_compile, mvn_test, test_path)
            else:
                # 进入到子目录当中
                target_PATH = os.path.join(excute_path, sub_project_name)
                os.chdir(target_PATH)
                write_cont, compile_success, test_result = self.Compile_Test_sub_unit(mvn_compile, mvn_test, test_path)
        os.chdir(current_dir)

        if compile_result == 0 and "[ERROR] COMPILATION ERROR :" not in write_cont: raise Exception(
            "Mvn execute failed")
        compile_logInfo_path = os.path.join(self.LogINFO_PATH, os.path.basename(test_file_name))
        with open(compile_logInfo_path, 'w', encoding='utf-8') as f:
            f.write(write_cont)

        # 处理执行mvn test 保存到 ./target/Surefire_reports/* 当中的信息
        Surefire_reports_dst_file = self.Surefire_reports_TEST_info(write_cont, test_file_name, Dtest_para)

        return compile_result, test_result, compile_logInfo_path, Surefire_reports_dst_file

    # 记录编译和测试结果和测试运行日志
    def Compile_Test_sub_unit(self, mvn_compile, mvn_test, test_path):
        compile_success, test_success = 0, 0
        compile_result = subprocess.run(mvn_compile, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                        universal_newlines=True, shell=True)
        write_cont = "original test path: " + test_path + "\n########## Compile INFO ##########\n" + compile_result.stdout + compile_result.stderr

        if "BUILD SUCCESS" in compile_result.stdout or "BUILD SUCCESS" in compile_result.stderr:
            compile_success = 1
            test_result = subprocess.run(mvn_test, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         universal_newlines=True, env=env, shell=True)
            write_cont = "original test path: " + test_path + "\n########## Compile INFO ##########\n" + compile_result.stdout + compile_result.stderr + \
                         "\n########## Test INFO ##########\n" + test_result.stdout + test_result.stderr

            if "BUILD SUCCESS" in test_result.stdout or "BUILD SUCCESS" in test_result.stderr:
                test_success = 1

        return write_cont, compile_success, test_success

    # 处理Surefire测试报告的信息
    def Surefire_reports_TEST_info(self, INFO_CONT, test_file_name, Dtest_para):
        # 生成xml文件，名称格式为TEST-<测试文件名>.xml
        file_name = "TEST-" + os.path.basename(test_file_name).replace(".java", ".xml")
        start_index = INFO_CONT.find("[ERROR] Please refer to ")
        if start_index < 0: return
        # 提取Surefire报告的路径，从[ERROR] Please refer to 之后到 for the individual test results.之前
        start_index = start_index + len("[ERROR] Please refer to ")
        end_index = INFO_CONT.find(" for the individual test results.")
        surefire_reports_PATH = INFO_CONT[start_index:end_index]
        # 如果surefire_reports_PATH路径存在，则将文件复制到指定的目录当中
        Surefire_reports_dst_file = ""
        if surefire_reports_PATH != "":
            surefire_reports_Name = "TEST-" + Dtest_para + ".xml"
            src_path = os.path.join(surefire_reports_PATH, surefire_reports_Name)
            shutil.copy(src_path, self.Surefire_reports_dest_path)      # 从源路径复制到目标路径
            # 将文件重命名， 防止被覆盖
            Surefire_reports_dst_file = os.path.join(self.Surefire_reports_dest_path, os.path.basename(src_path))
            os.rename(Surefire_reports_dst_file, os.path.join(self.Surefire_reports_dest_path, file_name))
            Surefire_reports_dst_file = os.path.join(self.Surefire_reports_dest_path, file_name)
        return Surefire_reports_dst_file


class Unit:
    def __init__(self, model_path) -> None:
        if "CodeLlama-34b-Instruct" in model_path:
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

            system_prompt = '''
            You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            '''
            system_prompt = f"{B_SYS}{system_prompt}{E_SYS}"
            self.problem_prompt = (system_prompt + "[INST] {instruction} [/INST]")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                              torch_dtype=torch.float16).cuda()

        elif "CodeFuse-CodeLlama" in model_path:
            HUMAN_ROLE_START_TAG = "<|role_start|>human<|role_end|>"
            BOT_ROLE_START_TAG = "<|role_start|>bot<|role_end|>"
            self.problem_prompt = (HUMAN_ROLE_START_TAG + "{instruction}" + BOT_ROLE_START_TAG)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                              torch_dtype=torch.float16).cuda()

    def generate(self, prompt):
        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(), max_new_tokens=1024, max_length=2048
        )
        output_ids = output_ids[0][len(input_ids[0]):]
        outputs = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return outputs

    # 根据输入的提示（ask_test_method_prompt）和是否需要修复（repair_TAG）来调用不同的模型生成测试方法
    def method_pred_unit(self, ask_test_method_prompt, repair_TAG=False):
        if repair_TAG:
            if "glm-zero-preview" in model_path:
                response_test = openai.ChatCompletion.create(
                    model=model_path,
                    messages=[
                        {"role": "system",
                         "content": "I want you to play the role of a professional who repairs buggy lines of the test method. Unnecessary import statement can be removed."},
                        {"role": "user", "content": ask_test_method_prompt},
                    ],
                    temperature=0)
                generated_content = response_test.choices[0].message['content']
            else:
                role = "I want you to play the role of a professional who repairs buggy lines of the test method."
                instruction = role + '\n\n' + ask_test_method_prompt
                prompt = self.problem_prompt.format(instruction=instruction)
                generated_content = self.generate(prompt)

        else:
            if "glm-zero-preview" in model_path:
                response_test = openai.ChatCompletion.create(
                    model=model_path,
                    messages=[
                        {"role": "system",
                         "content": "I want you to play the role of a professional who writes Java test method."},
                        {"role": "user", "content": ask_test_method_prompt},
                    ],
                    temperature=0)
                generated_content = response_test.choices[0].message['content']

            else:
                role = "I want you to play the role of a professional who writes Java test method for the Focal method. The following is the Class, Focal method and Import information."
                instruction = role + '\n\n' + ask_test_method_prompt
                prompt = self.problem_prompt.format(instruction=instruction)
                generated_content = self.generate(prompt)

        test_method, import_statement = self.return_code(generated_content)
        return test_method, import_statement

    # 从LLM生成的响应中提取测试代码和import导入语句
    def return_code(self, gen_cont):
        gen_cont = '\n'.join([line for line in gen_cont.split('\n') if "Below is " not in line])
        gen_cont = gen_cont.replace("(Fixed)","").replace("java\r\n","").replace("...","").replace("java\n","").replace("Java\n","").replace("# Test Method","")
        # find code

        pattern = r"```(.*?)```"
        matches = re.findall(pattern, gen_cont, re.DOTALL)
        matchCode = [match for match in matches if len(match) > 5 and " void " in match]
        if len(matchCode) == 0:
            matchCode = gen_cont
        else:
            matchCode = matchCode[-1]
        JavaCode_list = matchCode.split("\n")

        import_statements = []
        TAG = False
        for line_code in JavaCode_list:
            if "import " in line_code:
                TAG = True
                import_statements.append(line_code)
            elif TAG==True:
                break
        import_statement = "\n".join(import_statements)

        codeBlock = []
        left_brack_list = []
        right_brack_list = []
        Start_Tag = False
        for current_line_number, line in enumerate(JavaCode_list, start=1):
            if ("@Test" in line or " void " in line) and Start_Tag == False:
                Start_Tag = True
                if "@Test" not in line:  # 生成的代码当中可能没有 @Test 这个关键字
                    line_str = "@Test\n" + line
                    codeBlock.append(line_str)
                else:
                    codeBlock.append(line)

                left_brack_count = line.count("{")
                left_brack_list.extend(["{"] * left_brack_count)
                right_brack_count = line.count("}")
                right_brack_list.extend(["}"] * right_brack_count)
                continue
            if Start_Tag:
                codeBlock.append(line)

                left_brack_count = line.count("{")
                left_brack_list.extend(["{"] * left_brack_count)
                right_brack_count = line.count("}")
                right_brack_list.extend(["}"] * right_brack_count)
                if len(left_brack_list) == len(right_brack_list):
                    break
        codeBlock_str = "\n".join(codeBlock)

        return codeBlock_str, import_statement

    # 推断给定方法（focal_method_name）的总体意图
    # input：ori_test_Path, output：method intention
    def intention_unit(self, PL_Focal_Method, focal_method_name):
        if "glm-zero-preview" in model_path:
            Intention_NL = f'''Please describe the overall intention of the {focal_method_name} method in as much detail as possible in one sentence.'''
            # Intention_NL = f''Please infer the overall intention of the {focal_method_name} method with one sentence.
            ask_intention_prompt = PL_Focal_Method + '\n\n' + Intention_NL
            response_intention = openai.ChatCompletion.create(
                model=model_path,
                messages=[
                    {"role": "system",
                     "content": "I want you to play the role of a professional who infers method intention."},
                    {"role": "user", "content": ask_intention_prompt},
                ],
                temperature=0
            )
            intentions = response_intention.choices[0].message['content']
        else:
            role = "I want you to play the role of a professional who infers method intention."
            Intention_NL = f'Please tell me the intention of the {focal_method_name} method.'
            ask_intention_prompt = PL_Focal_Method + '\n\n' + Intention_NL
            instruction = role + '\n\n' + ask_intention_prompt
            prompt = self.problem_prompt.format(instruction=instruction)
            intentions = self.generate(prompt)

        intentions = " ".join(intentions.split())
        # intention = re.split('(?<=[.!?]) +', intentions)
        return intentions

    def commentDelete(self, java_code):
        """
        Removes all comments from the given Java code, while preserving the content
        within string literals.
        """
        # Pattern to match string literals (both single and double-quoted) or comments
        pattern = r'(".*?"|\'.*?\'|//.*?$|/\*.*?\*/)'

        def replace_func(match):
            # If the matched text is a comment (either single or double slash), remove it (replace with empty string)
            # Otherwise, it's a string literal, so we keep it
            if match.group(0).startswith(("//", "/*")):
                return ""  # Remove comments
            else:
                return match.group(0)  # Keep string literals

        # Using the sub function with a replacement function
        cleaned_code = re.sub(pattern, replace_func, java_code, flags=re.DOTALL | re.MULTILINE)
        return cleaned_code


if __name__ == "__main__":
    projects_name = ['sachin-handiekar_jInstagram.json', 'tabulapdf_tabula-java.json', 'Zappos_zappos-json.json']

    for project_name in projects_name:
        print("project_name: "+project_name)
        Json_file_Path = os.path.join(AgentTesterDir, "RepoData", project_name)
        ChatGptTester(project_name.replace(".json", ""))

        # Final Result postprocessing
        # ProceFinalResult(project_name.replace(".json", ""))
        ProceFinalResult(project_name)
