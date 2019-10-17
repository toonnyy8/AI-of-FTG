import "core-js/stable"
import "regenerator-runtime/runtime"

import { registerLayers } from "./layers"
export { registerLayers }

import { registerFuncs } from "./functions"
export { registerFuncs }

import { registerScope } from "./scope"

export { registerScope }

import { registerTool } from "./tool"

export { registerTool }

import { registerSL } from "./sl"

export { registerSL }

export const registerTfex = (tf) => {
    return {
        layers: registerLayers(tf),
        funcs: registerFuncs(tf),
        scope: registerScope(tf),
        tool: registerTool(tf),
        sl: registerSL(tf)
    }
}