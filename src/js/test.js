// import * as tf from "@tensorflow/tfjs"
import * as tf from "@tensorflow/tfjs/dist/tf"
tf

import { positionwiseFF } from "./MirageNet/transformerXL/model"
let a_ = new positionwiseFF({ axis: 0 })
a_.apply(tf.tensor([1, 2, 3]))
let a_2 = new positionwiseFF({ axis: 0 })
a_2.apply(tf.tensor([1, 2, 3]))