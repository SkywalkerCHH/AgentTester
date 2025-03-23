import os
from FileRemove import chatTester
current_dir = os.path.dirname(os.path.abspath(__file__))


# 从一个Java代码列表中提取出以@Test注解或void关键字开头的代码块，并确保代码块中的大括号 {} 是匹配的：
def return_code(JavaCode_list):
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
    codeBlock_str = "".join(codeBlock)
    return codeBlock_str


# 从给定的Java代码字符串中提取ESTest_scaffolding类的内容
def return_code_evosuiteFirst(JavaCode_str):
    javaCode = JavaCode_str.split("ESTest_scaffolding {", 1)[-1]    # 分割为两部分，取"ESTest_scaffolding {"之后的内容
    last_bracket_index = javaCode.rfind('}')    # rfind('}')：从右向左查找字符串中倒数第一个右大括号 } 的位置
    return javaCode[:last_bracket_index]    # 返回从字符串开头到最后一个右大括号}之间的部分，即ESTest_scaffolding类的内容


# 提取类定义块
def return_class(JavaCode_list):
    class_block = []
    for code_line in JavaCode_list:
        if "public class " in code_line and " extends " in code_line:   # 找到类定义
            class_block.append(code_line)
            break
        elif "import " in code_line:    # 忽略导入语句
            continue
        else:
            class_block.append(code_line)   # 将其他行添加到class_block
            continue
    classBlock_str = "".join(class_block) + "\nTOFILL\n" + "}\n"    # TOFILL是一个占位符，表示后续待填充其他内容，}表示类的结束括号
    return classBlock_str


def return_import(JavaCode_list):
    import_block = []
    for code_line in JavaCode_list:
        if "import " in code_line:
            import_block.append(code_line)
        else: continue
    importBlock_str = "".join(import_block)
    return importBlock_str


# 将指定目录下的多个Java测试文件合并为一个Java测试文件
def IterateStart(DIR, tag="evosuite_first"):
    java_files = [os.path.join(root, file) for root, dirs, files in os.walk(DIR) for file in files if file.endswith('.java') and "ESTest#" in file]
    # os.walk：遍历指定目录及其子目录，找到所有以.java结尾且包含"ESTest#"的文件。
    while java_files:
        className = os.path.basename(java_files[0]).split("#")[0]

        # 从java_files 中获取所有类名（className） 相同的.java 文件
        Deal_java_files = [file for file in java_files if os.path.basename(file).split("#")[0] == className]

        with open(Deal_java_files[0], 'r', encoding='utf-8') as f:
            JavaCode_list = f.readlines()
            classFrame_block_old = return_class(JavaCode_list)
        classFrame_block = classFrame_block_old.replace(os.path.basename(Deal_java_files[0]).replace(".java", "")+" extends", className + " extends")

        # 从 Deal_java_files 当中获取所有的测试方法和导入语句，并去重:
        Deal_allTestMethods = []
        Deal_allImports = []
        for Deal_file in Deal_java_files:
            with open(Deal_file, 'r', encoding='utf-8') as f:
                if tag == "evosuite_first":
                    testMethod = return_code_evosuiteFirst(f.read())
                else: testMethod = return_code(f.readlines())

                if testMethod not in Deal_allTestMethods: Deal_allTestMethods.append(testMethod)
                f.seek(0)
                Deal_allImports.append(return_import(f.readlines()))
        Deal_allImport ="\n".join(list(set(Deal_allImports)))   # 使用set将各Imports语句转换为一个集合，并进行去重处理

        Final_code = classFrame_block.replace("@RunWith(", Deal_allImport + "\n@RunWith(").replace("TOFILL", "\n".join(Deal_allTestMethods))
        print(Final_code)

        # 保存最终Java代码：
        new_filePath = os.path.join(os.path.dirname(Deal_java_files[0]), className+".java")
        with open(new_filePath, 'w', encoding='utf-8') as f:
            f.write(Final_code)

        # 将处理之后的元素删除
        for element in Deal_java_files:
            os.remove(element)
            while element in java_files:
                java_files.remove(element)


if __name__ == "__main__":
    import os

    ProjectResultPath = os.path.join(current_dir, 'ProjectResult')
    # WAYs = ['ChatTester', 'Evosuite_first', "ChatGPT"]
    WAYs = ['AgentTester']
    for way in WAYs:
        projects_name = ['sachin-handiekar_jInstagram', 'tabulapdf_tabula-java', 'Zappos_zappos-json']
        # projects_name = ['sachin-handiekar_jInstagram']

        for project_name in projects_name:
                ChatTester_result = chatTester(project_name, "IterateResultDeal")  # 先将文件移动到合适位置
                DIR = os.path.join(ProjectResultPath, way, project_name)  # 然后再进行合并
                IterateStart(DIR, tag='AgentTester')
