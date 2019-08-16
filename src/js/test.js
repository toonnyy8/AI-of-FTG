import * as tf from "@tensorflow/tfjs"
import { LayerNormalization } from "./MirageNet/transformerXL/model"
let a_ = new LayerNormalization({ axis: 0 })
a_.apply(tf.tensor([1, 2, 3]))
let a_2 = new LayerNormalization({ axis: 0 })
a_2.apply(tf.tensor([1, 2, 3]))