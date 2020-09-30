import * as tf from "@tensorflow/tfjs"
import * as nn from "./nn"


export const AED = (layers: number, inputChannels: number) => {
    const filters = new Array(layers)
        .fill(0)
        .map((_, idx) => 64 * 2 ** Math.floor(idx / 2))
        .reverse()
        .concat(inputChannels)
        .reverse()

    const convs = filters
        .slice(1)
        .map(f => tf
            .layers
            .separableConv2d({
                kernelSize: 3,
                filters: f,
                strides: 2,
                padding: "same"
            })
        )
    const deconvs = filters
        .slice(0, -1)
        .reverse()
        .map(f => tf
            .layers
            .separableConv2d({
                kernelSize: 3,
                filters: f,
                padding: "same"
            })
        )

    return {
        ae: (input: tf.Tensor3D | tf.Tensor4D) =>
            tf.tidy(() =>
                convs.reduce((inp, conv) =>
                    nn.mish(<tf.Tensor>conv.apply(inp)),
                    input,
                )
            ),
        ae_ws: () =>
            tf.tidy(() =>
                convs.reduce(
                    (w, conv) =>
                        w.concat(...<tf.Variable[]>conv.getWeights()),
                    <tf.Variable[]>[],
                )
            ),

        ad: (input: tf.Tensor3D | tf.Tensor4D) =>
            tf.tidy(() =>
                deconvs.reduce((inp, deconv) =>
                    nn.mish(<tf.Tensor>deconv.apply(nn.insertPadding(inp))),
                    input,
                )
            ),
        ad_ws: () =>
            tf.tidy(() =>
                deconvs.reduce(
                    (w, deconv) =>
                        w.concat(...<tf.Variable[]>deconv.getWeights()),
                    <tf.Variable[]>[],
                )
            ),
    }

}