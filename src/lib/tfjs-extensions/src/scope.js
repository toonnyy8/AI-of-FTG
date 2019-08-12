import * as tf from "@tensorflow/tfjs"

let tfexVariables = []

export function getVariable({ Name = "name", scopeName = "scopeName" || ["scope1Name", "scope2Name"], trainable = null || true || false }) {
    let returnVars = tfexVariables

    if (Name) {
        returnVars = returnVars.filter((value, index, obj) => {
            return value.name.split("/").slice(-1) == Name
        })
    }

    if (scopeName) {
        if (Array.isArray(scopeName)) {
            returnVars = returnVars.filter((value, index, obj) => {
                let scopes = value.name.split("/")
                let scopeTrue = true
                for (let i = 0; i < scopeName.length && scopeTrue; i++) {
                    scopeTrue = scopes[i] == scopeName[i]
                }
                return scopeTrue
            })
        } else {
            returnVars = returnVars.filter((value, index, obj) => {
                return !!value.name.split("/").find((value1, index1, obj1) => {
                    return value1 == scopeName
                })
            })
        }
    }

    if (trainable != null && trainable != undefined) {
        returnVars = returnVars.filter((value, index, obj) => {
            return value.trainable == trainable
        })
    }
    return returnVars
}

export function variableScope(scopeName = "scopeName" || ["scope1Name", "scope2Name"], variables = [tf.variable()] || null, joinScope = true, autoCheck = false) {
    if (variables != [] && variables != null && variables != undefined) {
        variables.map((x) => {
            let tempOldScope = x.name.split("/")
            let tempName = ""
            for (let i = 0; i < tempOldScope.length - 1; i++) {
                tempName += `${tempOldScope[i]}/`
            }

            if (Array.isArray(scopeName)) {
                x.name = `${tempName}${scopeName.join("/")}/${tempOldScope.slice(-1)}`
            } else {
                x.name = `${tempName}${scopeName}/${tempOldScope.slice(-1)}`
            }

            if (joinScope) {
                if (!autoCheck || !tfexVariables.find((value, index, obj) => {
                    return value === x
                })) {
                    tfexVariables.push(x)
                }
            }

            return x
        })
    }
    return (f = (Vars = [tf.variable()]) => { }) => {
        for (let i = 0; i < tfexVariables.length; i++) {
            tfexVariables[i].trainable = false
        }

        let Vars = getVariable({ Name: null, scopeName: scopeName, trainable: null })
        for (let i = 0; i < Vars.length; i++) {
            Vars[i].trainable = true
        }

        f(Vars)

        for (let i = 0; i < tfexVariables.length; i++) {
            tfexVariables[i].trainable = true
        }
    }
}