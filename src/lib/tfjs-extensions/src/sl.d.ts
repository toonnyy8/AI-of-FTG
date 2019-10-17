import * as tf from "@tensorflow/tfjs"

declare function save(tensorList: { [key: string]: tf.Tensor }): Uint8Array
declare function load(saveTensorList: Uint8Array): { [key: string]: tf.Tensor }

declare function registerSL(tf): {
    save(tensorList: { [key: string]: tf.Tensor }): Uint8Array
    load(saveTensorList: Uint8Array): { [key: string]: tf.Tensor }
}