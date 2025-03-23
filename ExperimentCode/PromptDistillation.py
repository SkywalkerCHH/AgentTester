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
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

"""
提示蒸馏器（PD，Prompt Distillation）:
利用 LLM 多温度采样生成多个候选提示及其对应的测试用例，然后根据测试用例的编译结果选出有效的测试用例作为下一阶段的少样本提示。
"""

# TODO modify this path
java_home = "C:/Program Files/Java/jdk1.8.0_152"
os.environ["JAVA_HOME"] = java_home
env = os.environ.copy()
env['JAVA_TOOL_OPTIONS'] = '-Duser.language=en -Duser.country=US'

current_dir = os.path.dirname(os.path.abspath(__file__))  # 返回该脚本的绝对路径，D:\AgentTester v2\ExperimentCode
AgentTesterDir = os.path.dirname(current_dir)    # 返回父目录，即 D:\AgentTester v2
testedRepo_PATH = os.path.join(AgentTesterDir, "Repos")  # 存放repo的 path，即 D:\AgentTester v2\Repos

# model_path = "glm-4"
model_path = "glm-zero-preview"
# model_path = "deepseek-r1:8b"

class PromptDistillation:
    def __init__(self, Intention_TAG):
        self.Intention_TAG = Intention_TAG

        if "CodeLlama-34b-Instruct" in model_path:
            sub_save_dir = "CodeLlama"
        elif "CodeFuse-CodeLlama" in model_path:
            sub_save_dir = "CodeFuse"
        elif "glm-zero-preview" in model_path:
            sub_save_dir = os.path.basename(Json_file_Path).replace(".json", "")
            # TODO set api_key
            openai.api_base = ""
            openai.api_key = ""

        self.original_java_PATH = os.path.join(current_dir, self.Intention_TAG, sub_save_dir, 'original_java')
        self.LogINFO_PATH = os.path.join(current_dir, self.Intention_TAG, sub_save_dir, 'LogINFO')
        self.Surefire_reports_dest_path = os.path.join(current_dir, self.Intention_TAG, sub_save_dir, 'Surefire_reports')
        self.GeneratedTest_PATH = os.path.join(current_dir, self.Intention_TAG, sub_save_dir, 'GeneratedTest')
        self.GeneratedTest_PATH_2 = os.path.join(current_dir, self.Intention_TAG, sub_save_dir, 'GeneratedTest_2')
        self.MetricOut_Path = os.path.join(current_dir, self.Intention_TAG, sub_save_dir, 'result_1.json')
        self.result_2_Path = os.path.join(current_dir, self.Intention_TAG, sub_save_dir, 'result_2.json')
        self.Composit_prompt_PATH = os.path.join(current_dir, self.Intention_TAG, sub_save_dir, 'Composit_prompt')

        self.boolean(self.GeneratedTest_PATH_2)    # 初始化文件夹
        self.unit_instance = LLM_Unit(model_path)  # LLM-class instance

        self.get_initial_prompt_path(self.MetricOut_Path)

    # 创建/清空文件夹
    def boolean(self, file_path):
        if not os.path.exists(file_path):  # 检查路径是否存在
            print('Creat folder....')
            os.makedirs(file_path)  # 不存在则创建路径
        else:
            shutil.rmtree(file_path)  # 存在则删除路径
            os.makedirs(file_path)  # 重新创建路径

    # 1.从result_1.json中读取编译失败的测试路径，获取其初始提示
    # 2.生成候选提示，获取对应测试代码，并运行
    def get_initial_prompt_path(self, result_1):
        with open(result_1, 'r') as f:
            lines = json.load(f)
            for line in tqdm(lines):
                original_path = line.get('original_path')
                generated_path = line.get('generated_path')
                Compile = line.get('Compile')
                Test = line.get('Test')

                if Test == 1:  # 编译成功，将generated_path文件复制到GeneratedTest_PATH_2文件夹
                    self.write_result_2(original_path, generated_path, Compile, Test)

                else:   # 编译失败，则获取其对应的初始提示路径
                    # 将生成的测试路径分割为目录列表
                    dirs = generated_path.split(os.sep)
                    # 替换为初始提示的文件目录名
                    dirs[dirs.index('GeneratedTest')] = 'Composit_prompt'
                    # 组合为初始提示路径
                    initial_prompt_path = os.sep.join(dirs)
                    temperature = [0.3, 0.6, 0.8, 1]
                    # temperature = [0.3, 0.6]
                    max_count = 4
                    count = 0
                    for temp in temperature:
                        count += 1
                        print('\n########## 第{}轮 ###########'.format(count))
                        focal_method_name, test_method, import_statement = self.genCandidatePrompt(initial_prompt_path, temp)
                        Test = self.gen_test_2(Json_file_Path, original_path, focal_method_name, test_method, import_statement)
                        # 编译成功则退出循环，否则用最多4个不同temp生成候选提示
                        if Test == 1:
                            print('\n~~~~~编译测试成功！~~~~~\n')
                            break
                        elif count == max_count and Test == 0:
                            print('\n-----编译测试失败！-----\n')
                            self.write_result_2(original_path, generated_path, Compile, Test)

        return

    # 根据初始提示生成候选提示CandidatePrompt,获取测试代码
    def genCandidatePrompt(self, initial_prompt_path, temp):
        with open(initial_prompt_path, 'r', encoding='utf-8') as f:
            initial_prompt = f.read()  # 读取初始提示内容
            focal_method_name = os.path.basename(initial_prompt_path).split('#')[1].split('.')[0]  # 获取被测试方法名称

            new_method_intention, new_instruction = self.unit_instance.CandidatePrompt(initial_prompt, focal_method_name, temp)
            parts = initial_prompt.split("# Method intention")
            parts[1] = "# Method intention\n" + new_method_intention + "\n\n" + parts[1].split("# Instruction")[0]
            parts_2 = "# Instruction\n" + new_instruction
            # 重新拼接复合提示
            CandidatePrompt = parts[0] + parts[1] + parts_2
            print('\n---------候选提示---------\n' + CandidatePrompt)
            test_method, import_statement = self.unit_instance.gen_test_method(CandidatePrompt)
        return focal_method_name, test_method, import_statement

    # 读取 RepoData 文件夹中的JSON文件，并基于文件中的信息处理代码、生成测试代码、执行测试并记录结果。
    def gen_test_2(self, Json_file_Path, original_path, focal_method_name, test_method, import_statement):
        project_name = os.path.basename(Json_file_Path).replace(".json", "")    # 获取文件路径中的文件名称
        with open(Json_file_Path, 'r', encoding='utf-8') as f:
            file_cont = json.load(f)    # 使用json.load函数读取JSON文件内容。

        for cont in file_cont:    # 遍历JSON文件中的每个条目，提取测试方法和被测试方法的相关信息。
            Under_test_method = cont['Under_test_method']
            Test_method = cont['Test_method']
            if len(Under_test_method) == 0: continue

            Method_statement = Under_test_method['Method_statement']
            TestInfo = Test_method['TestInfo']

            if Method_statement == focal_method_name and TestInfo == original_path:
                TestFileName = os.path.basename(Test_method['TestInfo'].split("###")[0])
                TestDir = os.path.dirname(Under_test_method['project_path'].split("###")[0].replace("/main/", "/test/"))
                TestFilePath = os.path.join(TestDir, TestFileName)
                TestScaffoldPath = os.path.join(TestDir,TestFileName.replace(".java","_scaffolding.java"))
                ScaffoldingCode = Test_method["scaffoldingCode"]
                TestCodeShell = Test_method['TestCodeShell']
                try:
                    self.boolean(TestDir)
                    with open(TestScaffoldPath, 'w', encoding='utf-8') as f:
                        f.write(ScaffoldingCode)

                    # 将生成的 test_method 写入 GeneratedTest_2，和 project 当中
                    Gen_TestfilePath, Dtest_para = self.file_write(test_method, TestFilePath, TestCodeShell, import_statement, focal_method_name)

                    # 执行测试，并获取编译和测试结果。junit版本设为4
                    compile_result, test_result, proced_compile_INFO, proced_test_INFO = self.adhoc_excute(Dtest_para, Gen_TestfilePath, TestFilePath, testedRepo_PATH, project_name, 4)

                    if test_result == 1:
                        out_dict = {"original_path": Test_method['TestInfo'],
                                    "generated_path": Gen_TestfilePath,
                                    "Compile": compile_result,
                                    "Test": test_result
                                    }
                        # 写入Contain_intention\.\result_2.json
                        try:
                            with open(self.result_2_Path, "r", encoding="utf-8") as f:
                                result2_data = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            result2_data = []
                        # 更新result_2
                        result2_data.append(out_dict)
                        with open(self.result_2_Path, "w", encoding="utf-8") as f:
                            json.dump(result2_data, f, indent=4)  # 记录结果到JSON文件
                            # f.write("\n")

                except Exception as e:
                    traceback.print_exc()
                    test_result = 0

                finally:
                    # 删除repos中的test文件
                    test_path = os.path.join(testedRepo_PATH, project_name, "src", "test")
                    self.boolean(test_path)
                    print('Project reset.')
                    return test_result

    # 生成测试文件 GeneratedTest_2
    def file_write(self, test_method, TestFilePath, TestCodeShell,  Test_Import_info, focal_method_name):
        # 提取包名和类名：
        package_name = [code for code in TestCodeShell.split("\n") if "package " in code and ";" in code][0].replace("package ", "").replace(";", "").strip()
        class_name = os.path.basename(TestFilePath).replace(".java", "")
        # 将字符串中的特定部分替换为新的内容：
        codeShell_1 = TestCodeShell.replace("\nimport ", Test_Import_info + "\nimport ", 1)
        codeShell_2 = codeShell_1.replace("//TOFILLL", test_method)
        codeShell = "// original test path: " + TestFilePath + "\n" + codeShell_2

        # 生成测试文件路径:
        Gen_TestfilePath = os.path.join(self.GeneratedTest_PATH_2, os.path.basename(TestFilePath).replace(".java","#"+focal_method_name+".java"))
        with open(Gen_TestfilePath, 'w', encoding='utf-8') as f:
            f.write(codeShell)

        # 将 testFile 也放入 project 当中
        # self.boolean(TestFilePath)
        with open(TestFilePath, 'w', encoding='utf-8') as f:
            f.write(codeShell)

        return Gen_TestfilePath, package_name+"."+class_name

    def adhoc_excute(self, Dtest_para, Gen_TestfilePath, TestFilePath, testedRepo_PATH, project_name, JUNIT_VERSION):
        # 该函数用于执行Maven构建和测试任务，并生成相应的日志和测试报告
        excute_path = os.path.join(testedRepo_PATH, project_name)  # 将testedRepo_PATH和project_name拼接成项目的执行路径
        os.chdir(excute_path)

        mvn_compile = ['mvn', 'test-compile', '-Dcheckstyle.skip=true']
        mvn_test = ['mvn', 'test', '-Dcheckstyle.skip=true']
        if JUNIT_VERSION == 5:
            mvn_compile = ['mvn', 'test-compile', '-Dtest.engine=junit-jupiter', '-Dcheckstyle.skip=true']
            mvn_test = ['mvn', 'test', '-Dtest.engine=junit-jupiter', '-Dcheckstyle.skip=true']

        write_cont, compile_result, test_result = self.Compile_Test_sub_unit(mvn_compile, mvn_test, TestFilePath)

        # 未能正确的执行mvn 指令。此时首先需要执行 mvn clean
        if compile_result != 1 and "[ERROR] COMPILATION ERROR :" not in write_cont and "Could not resolve dependenci" in write_cont:
            mvn_install = ['mvn', 'clean', 'install']
            mvn_result = subprocess.run(mvn_install, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                        universal_newlines=True, shell=True)   # Windows系统要加上shell=True！！
            if "BUILD SUCCESS" in mvn_result.stdout or "BUILD SUCCESS" in mvn_result.stderr:
                write_cont, compile_result, test_result = self.Compile_Test_sub_unit(mvn_compile, mvn_test, TestFilePath)
        os.chdir(current_dir)

        if compile_result == 0 and "[ERROR] COMPILATION ERROR :" not in write_cont: raise Exception("Mvn execute failed")
        compile_logInfo_path = os.path.join(self.LogINFO_PATH, os.path.basename(Gen_TestfilePath))
        with open(compile_logInfo_path, 'w', encoding='utf-8') as f:
            f.write(write_cont)

        # 处理执行mvn test 保存到 ./target/Surefire_reports/* 当中的信息
        Surefire_reports_dst_file = self.Surefire_reports_TEST_info(write_cont, os.path.basename(Gen_TestfilePath), Dtest_para)

        return compile_result, test_result, compile_logInfo_path, Surefire_reports_dst_file

    # 编译和测试一个Maven项目：
    def Compile_Test_sub_unit(self, mvn_compile, mvn_test, test_path):
        compile_success, test_success = 0, 0
        compile_result = subprocess.run(mvn_compile, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                        universal_newlines=True, shell=True)
        write_cont = "original test path: " + test_path + "\n########## Compile INFO ##########\n" + compile_result.stdout + compile_result.stderr

        if "BUILD SUCCESS" in compile_result.stdout or "BUILD SUCCESS" in compile_result.stderr:
            compile_success = 1
            # 使用subprocess.run方法执行Maven测试命令：
            test_result = subprocess.run(mvn_test, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=env, shell=True)
            # 将测试路径、编译结果和测试结果（包括标准输出和标准错误）拼接成一个字符串：
            write_cont = "original test path: " + test_path + "\n########## Compile INFO ##########\n" + compile_result.stdout + compile_result.stderr + \
                         "\n########## Test INFO ##########\n" + test_result.stdout + test_result.stderr

            if "BUILD SUCCESS" in test_result.stdout or "BUILD SUCCESS" in test_result.stderr:
                test_success = 1

        return write_cont, compile_success, test_success

    # 处理Surefire测试报告的信息
    def Surefire_reports_TEST_info(self, INFO_CONT, test_file_name, Dtest_para):
        # Surefire是Maven中的一个插件，用于运行和报告JUnit测试
        file_name = "TEST-" + os.path.basename(test_file_name).replace(".java", ".xml")
        start_index = INFO_CONT.find("[ERROR] Please refer to ")
        if start_index < 0: return
        start_index = start_index + len("[ERROR] Please refer to ")
        end_index = INFO_CONT.find(" for the individual test results.")
        surefire_reports_PATH = INFO_CONT[start_index:end_index]
        Surefire_reports_dst_file = ""
        if surefire_reports_PATH != "":
            surefire_reports_Name = "TEST-" + Dtest_para + ".xml"
            src_path = os.path.join(surefire_reports_PATH, surefire_reports_Name)
            shutil.copy(src_path, self.Surefire_reports_dest_path)
            Surefire_reports_dst_file = os.path.join(self.Surefire_reports_dest_path, os.path.basename(src_path))
            # 检查目标文件是否已存在，如果存在则删除
            dst_path = os.path.join(self.Surefire_reports_dest_path, file_name)
            if os.path.exists(dst_path):
                os.remove(dst_path)
            # 重命名文件
            os.rename(Surefire_reports_dst_file, dst_path)
        return Surefire_reports_dst_file

    # 将原编译结果复制至 result_2.json；同时复制原测试文件至 GeneratedTest_2
    def write_result_2(self, original_path, generated_path, Compile, Test):
        generated_path_2 = os.path.join(self.GeneratedTest_PATH_2, os.path.basename(generated_path))
        shutil.copyfile(generated_path, generated_path_2)
        out_dict = {"original_path": original_path,
                    "generated_path": generated_path_2,
                    "Compile": Compile,
                    "Test": Test
                    }
        # 写入result_2.json
        try:
            with open(self.result_2_Path, "r", encoding="utf-8") as f:
                result2_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            result2_data = []
        # 添加新的数据
        result2_data.append(out_dict)
        with open(self.result_2_Path, "w", encoding="utf-8") as f:
            json.dump(result2_data, f, indent=4)  # 记录结果到JSON文件
            # f.write("\n")


class LLM_Unit:
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

    def generate(self, prompt):  # 生成基于给定提示的文本
        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(), max_new_tokens=1024, max_length=2048
        )
        output_ids = output_ids[0][len(input_ids[0]):]
        outputs = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return outputs

    # LLM 根据候选提示生成测试
    def gen_test_method(self, CandidatePrompt):  # 生成测试方法
        if "glm-zero-preview" in model_path:   # 向LLM发送指令，要求根据代码及其意图生成测试
            response_test = openai.ChatCompletion.create(
                model=model_path,
                messages=[
                    {"role": "system",
                     "content": "I want you to play the role of a professional who writes Java test method using JUnit 4.Please ensure adequate coverage,and one testcase is enough."},
                    {"role": "user", "content": CandidatePrompt},
                ],
                temperature=0)
            full_generated_content = response_test.choices[0].message['content']
            generated_content = re.sub(r'<think>.*?</think>', '', full_generated_content, flags=re.DOTALL)  #删除<think>内容
        else:
            role = "I want you to play the role of a professional who writes Java test method for the Focal method. The following is the Class, Focal method and Import information."
            instruction = role + '\n\n' + CandidatePrompt
            prompt = self.problem_prompt.format(instruction=instruction)
            generated_content = self.generate(prompt)

        # print('LLM生成测试响应1：\n', generated_content)
        test_method, import_statement = self.return_code(generated_content)
        return test_method, import_statement

    # LLM 生成候选提示
    def CandidatePrompt(self, Composit_prompt, focal_method_name, temp):
        original_contents = "Original Contents:\n"
        CandidatePrompt_NL = f'''According to the original contents, please regenerate a more concise and perfect method intention('# Method intention'part ) and a more professional、comprehensive and concise test generation instruction('# Instruction'part) for {focal_method_name}.No need to generate test,only provide the Output.\n\n'''
        output = "Output:\n# New Method intention\n{new method intention}\n\n# New Instruction\n{new test generation instruction}\n"
        ask_intention_prompt = original_contents + Composit_prompt + '\n\n' + CandidatePrompt_NL + output
        response = openai.ChatCompletion.create(
            model=model_path,
            messages=[
                {"role": "system",
                 "content": "I want you to play the role of a professional LLM Prompt Engineer."},
                {"role": "user",
                 "content": ask_intention_prompt},
            ],
            temperature=temp
        )
        full_response = response.choices[0].message['content']
        response_new = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)  #删除<think>内容
        response_new = " ".join(response_new.split())  # 去除多余空格
        print('\nLLM 生成新指令：\n', response_new)
        new_method_intention, new_instruction = self.return_new(response_new)
        new_instruction = new_instruction + "\nThe generated code should be enclosed within ``` ```."
        return new_method_intention, new_instruction

    # 从 response_new 中获取 new method intention 和 new test generation instruction
    def return_new(self, response_new):
        parts = response_new.split("New Instruction")
        # 获取 Method intention 和 Instruction 的内容
        new_method_intention = parts[0].strip().replace("New Method intention", "").replace("#", "")
        new_instruction = parts[1].strip()
        return new_method_intention, new_instruction

    # 从 LLM 生成的测试中提取 Java测试代码 及其 import语句
    def return_code(self, gen_cont):
        gen_cont = '\n'.join([line for line in gen_cont.split('\n') if "Below is " not in line])  # 过滤掉包含"Below is "的行
        gen_cont = gen_cont.replace("(Fixed)", "").replace("java\r\n", "").replace("...", "").replace("java\n", "").replace("Java\n", "")
        print('\nLLM 生成测试：\n' + gen_cont)
        pattern = r"```(.*?)```"  # 查找被三个反引号（```）包围的代码块
        matches = re.findall(pattern, gen_cont, re.DOTALL)
        matchCode = [match for match in matches if len(match) > 5 and " void " in match][-1]  # 从中选择包含void的代码块
        JavaCode_list = matchCode.split("\n")

        # 从一段Java代码中提取出所有的import语句：
        import_statements = []
        TAG = False
        for line_code in JavaCode_list:
            if "import " in line_code:
                TAG = True
                import_statements.append(line_code)
            elif TAG==True:
                break
        import_statement = "\n".join(import_statements)

        # 一个Java代码列表中提取出以 @Test注解或 void关键字 开头的代码块，并确保代码块中的大括号 {} 是成对出现的：
        codeBlock = []              # 存储提取出的代码行
        left_brack_list = []
        right_brack_list = []       # 记录代码块中大括号的 左{ 和 右} 的数量
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

    # 从代码中删除注释
    def commentDelete(self, code):
        # comment delete
        regex = r"/\*(.|\\n)*?\*/"
        noMultilineComments = re.sub(regex, "", code)   # re.sub函数将匹配到的多行注释替换为空字符串

        # 去除单行注释，remove single line comments (// ...)：
        regex = r"//.*"
        non_comment_code = re.sub(regex, "", noMultilineComments)
        # 去除多行注释
        pattern = re.compile(r"(?s)/\*.*?\*/|//.*?[\r\n]")  # 用正则表达式匹配 /**...*/ 样式的注释
        codeWithoutComment = pattern.sub("", non_comment_code)

        return codeWithoutComment


if __name__ == "__main__":
    Intention_TAG = True
    if Intention_TAG:
        Intention = 'Contain_intention'
    else:
        Intention = "No_intention"

    projects_name = ['sachin-handiekar_jInstagram', 'tabulapdf_tabula-java', 'Zappos_zappos-json']

    for project_name in projects_name:
        print("project_name: "+project_name)
        Json_file_Path = os.path.join(AgentTesterDir, "RepoData", project_name + ".json")  # RepoData中各Json文件路径
        PromptDistillation(Intention)
