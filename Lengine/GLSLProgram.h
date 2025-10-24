#pragma once

#include <string>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace Lengine {


    class GLSLProgram
    {

    public:
        GLSLProgram();
        ~GLSLProgram();

        void compileShaders(const std::string& vertexShaderFilePath, const std::string& fragmentShaderFilePath);
        void linkShaders();
        void addAtrribute(const std::string& attributeName);

        GLint getUnifromLocation(const std::string& uniformName);

        void use();
        void unuse();
        void setMat4(const std::string& name, const glm::mat4& mat);
        void GLSLProgram::setMat3(const std::string& name, const glm::mat3& mat);
        void GLSLProgram::setVec3(const std::string& name, const glm::vec3& vec);
        void GLSLProgram::setVec4(const std::string& name, const glm::vec4& vec);

		GLuint getProgramID() { return _programID; }
    private:

        void compileShader(const std::string& filePath, GLuint id);

        GLuint _numAttributes;

		GLuint _programID;
        GLuint _vertexShaderID;
        GLuint _fragmentShaderID;


    };
}



