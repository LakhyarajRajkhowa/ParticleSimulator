#include "../Lengine/GLSLProgram.h"
#include "../Lengine/Errors.h"
#include <fstream>
#include <vector>


namespace Lengine {

    GLSLProgram::GLSLProgram() : _numAttributes(0), _programID(0), _vertexShaderID(0), _fragmentShaderID(0)
    {
    }

    GLSLProgram::~GLSLProgram()
    {
    }

    void GLSLProgram::compileShaders(const std::string& vertexShaderFilePath, const std::string& fragmentShaderFilePath)
    {


        _vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
        if (!_vertexShaderID)
        {
            fatalError("Vertex shader failed to be created!");
        }
        _fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
        if (!_fragmentShaderID)
        {
            fatalError("Fragment shader failed to be created!");
        }

        compileShader(vertexShaderFilePath, _vertexShaderID);
        compileShader(fragmentShaderFilePath, _fragmentShaderID);
    }
    void GLSLProgram::linkShaders()
    {

        // Attach our shaders to our program
        glAttachShader(_programID, _vertexShaderID);
        glAttachShader(_programID, _fragmentShaderID);

        // Link our program
        glLinkProgram(_programID);

        // Note the different functions here: glGetProgram* instead of glGetShader*.
        GLint isLinked = 0;
        glGetProgramiv(_programID, GL_LINK_STATUS, (int*)&isLinked);
        if (isLinked == GL_FALSE)
        {
            GLint maxLength = 0;
            glGetProgramiv(_programID, GL_INFO_LOG_LENGTH, &maxLength);

            // The maxLength includes the NULL character
            std::vector<GLchar> errorLog(maxLength);
            glGetProgramInfoLog(_programID, maxLength, &maxLength, &errorLog[0]);

            // We don't need the program anymore.
            glDeleteProgram(_programID);
            // Don't leak shaders either.
            glDeleteShader(_vertexShaderID);
            glDeleteShader(_fragmentShaderID);

            printf("%s\n", &(errorLog[0]));
            fatalError("Shaders  failed to link");
        }

        // Always detach shaders after a successful link.
        glDetachShader(_programID, _vertexShaderID);
        glDetachShader(_programID, _fragmentShaderID);
        glDeleteShader(_vertexShaderID);
        glDeleteShader(_fragmentShaderID);
    }

    void GLSLProgram::addAtrribute(const std::string& attributeName)
    {

        glBindAttribLocation(_programID, _numAttributes++, attributeName.c_str());
    }

    GLint GLSLProgram::getUnifromLocation(const std::string& uniformName) {
        GLuint location = glGetUniformLocation(_programID, uniformName.c_str());
        if (location == GL_INVALID_INDEX) {
            fatalError("Unifrom " + uniformName + " not found in shader!");
        }

        return location;

    }

    void GLSLProgram::use()
    {
        glUseProgram(_programID);
        /*for (int i = 0; i < _numAttributes; i++)
        {
            glEnableVertexAttribArray(i);
        }*/
    }
    void GLSLProgram::unuse()
    {
        glUseProgram(0);
       /* for (int i = 0; i < _numAttributes; i++)
        {
            glDisableVertexAttribArray(i);
        }*/
    }
    void GLSLProgram::compileShader(const std::string& filePath, GLuint id)
    {
        _programID = glCreateProgram();
        std::ifstream shaderFile(filePath);
        if (shaderFile.fail())
        {
            perror(filePath.c_str());
            fatalError("Error to open " + filePath);
        }

        std::string fileContents = "";
        std::string line;

        while (std::getline(shaderFile, line))
        {
            fileContents += line + "\n";
        }

        shaderFile.close();

        const char* contentsPtr = fileContents.c_str();
        glShaderSource(id, 1, &contentsPtr, nullptr);

        glCompileShader(id);

        GLint success = 0;
        glGetShaderiv(id, GL_COMPILE_STATUS, &success);

        if (success == GL_FALSE)
        {
            GLint maxLength = 0;
            glGetShaderiv(id, GL_INFO_LOG_LENGTH, &maxLength);

            // The maxLength includes the NULL character
            std::vector<GLchar> errorLog(maxLength);
            glGetShaderInfoLog(id, maxLength, &maxLength, &errorLog[0]);

            // Provide the infolog in whatever manor you deem best.
            // Exit with failure.
            glDeleteShader(id); // Don't leak the shader.

            printf("%s\n", &(errorLog[0]));
            fatalError("Shader " + filePath + " failed to compile");


        }

    }

    void GLSLProgram::setMat4(const std::string& name, const glm::mat4& mat) {
        glUniformMatrix4fv(glGetUniformLocation(_programID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
    }
    void GLSLProgram::setMat3(const std::string& name, const glm::mat3& mat) {
        GLint loc = glGetUniformLocation(_programID, name.c_str());
        if (loc == -1) {
            // Uniform not found
           fatalError("Warning: uniform '" + name + "' not found!\n");
            return;
        }
        glUniformMatrix3fv(loc, 1, GL_FALSE, glm::value_ptr(mat));
    }
    void GLSLProgram::setVec3(const std::string& name, const glm::vec3& vec) {
        GLint loc = glGetUniformLocation(_programID, name.c_str());
        if (loc == -1) return;  // uniform not found
        glUniform3f(loc, vec.x, vec.y, vec.z);
    }
    void GLSLProgram::setVec4(const std::string& name, const glm::vec4& vec) {
        GLint loc = glGetUniformLocation(_programID, name.c_str());
        if (loc == -1) return;  // uniform not found
        glUniform4f(loc, vec.r, vec.g, vec.b, vec.a);
    }
}



