import * as layers from "./layers"
export { layers }

export * from "./function"

import {VariableScope} from "./scope"

export let scope = new VariableScope("")