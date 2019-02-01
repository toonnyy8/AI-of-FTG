import * as gpuJs from 'gpu.js'


global.tempBuffer = {};

(function (exports) {
    //API兼容處理
    exports.URL = exports.URL || exports.webkitURL

    exports.requestAnimationFrame = exports.requestAnimationFrame ||
        exports.webkitRequestAnimationFrame || exports.mozRequestAnimationFrame ||
        exports.msRequestAnimationFrame || exports.oRequestAnimationFrame

    exports.cancelAnimationFrame = exports.cancelAnimationFrame ||
        exports.webkitCancelAnimationFrame || exports.mozCancelAnimationFrame ||
        exports.msCancelAnimationFrame || exports.oCancelAnimationFrame

    var isRecoding = false
    //預覽影片，使用原始的dom物件操作
    var video = document.createElement("video")
    var videoWidth = 192 * 2
    var videoHeight = 108 * 2
    video.autoplay = true
    video.height = videoHeight
    video.width = videoWidth

    //畫面展示需要，使用jquery dom 
    var dRecordBtn = document.getElementById("recodBtn")
    var dStopBtn = document.getElementById("stopBtn")
    dStopBtn.style = "display:none"

    let preFrame

    //錄制畫面用的canvas
    var canvas = document.getElementById('canvas')
    canvas.height = videoHeight
    canvas.width = videoWidth

    var ctx = canvas.getContext('2d')
    ctx.fillRect(0, 0, videoWidth, videoHeight)
    //灰階canvas
    var grayCanvas = document.getElementById('gray')
    grayCanvas.height = videoHeight
    grayCanvas.width = videoWidth

    var gctx = grayCanvas.getContext('2d')

    let gpu = new GPU()

    const toGray = function (a) {
        let reA = new Array()
        for (var x = 0; x < a.width; x++) {
            for (var y = 0; y < a.height; y++) {

                // Index of the pixel in the array  
                var idx = (x + y * a.width) * 4;
                var r = a.data[idx + 0];
                var g = a.data[idx + 1];
                var b = a.data[idx + 2];

                // calculate gray scale value  
                var gray = .299 * r + .587 * g + .114 * b;

                // assign gray scale value  
                reA.push(gray) // Red channel  
            }
        }

        return reA
    }

    const bufferDif = gpu.createKernel(function (a, b) {
        return Math.abs(a[this.thread.y][this.thread.x] - b[this.thread.y][this.thread.x])

    }).setOutput([videoWidth, videoHeight])

    const overlap = gpu.createKernel(function (a, b) {
        return Math.max(a[this.thread.y][this.thread.x], b[this.thread.y][this.thread.x] * 0.9)

    }).setOutput([videoWidth, videoHeight])
    //準備用來存放 requestAnimationFrame 的id 以便在停止時取消Canvas的截錄繪制
    var rafId = null
    //準備畢來存放，cnavas的錄制相關的物件
    var cStream = null
    var recorder = null
    var chunks = []
    //串流的來源
    var sourceTrack = null

    //開始錄制的邏輯
    function record(stream) {
        //將live區塊的影片來源跟串流接上。
        video.srcObject = stream
        shareStream = stream

        sourceTrack = stream.getTracks()[0]

        let grayBuffer = new Array()
        let workingFrame = true

        function drawVideoFrame(time) {
            rafId = requestAnimationFrame(drawVideoFrame)
            if (workingFrame) {
                ctx.drawImage(video, 0, 0, videoWidth, videoHeight)

                grayBuffer.push(new Float32Array(toGray(ctx.getImageData(0, 0, videoWidth, videoHeight))))

                let temp_ = bufferDif(gpuJs.input(grayBuffer.shift(), [videoWidth, videoHeight]), gpuJs.input(grayBuffer[0], [videoWidth, videoHeight]))
                global.tempBuffer = overlap(gpuJs.input(global.tempBuffer, [videoWidth, videoHeight]), gpuJs.input(temp_, [videoWidth, videoHeight]))
            }
            workingFrame = !workingFrame
        }
        //開始截取畫面並把requestAnimationFrame的id儲存起來以便控制

        cStream = canvas.captureStream(30)
        recorder = new MediaRecorder(cStream)

        grayBuffer.push(new Float32Array(toGray(ctx.getImageData(0, 0, videoWidth, videoHeight))))

        global.tempBuffer = new Float32Array(toGray(ctx.getImageData(0, 0, videoWidth, videoHeight)))

        rafId = requestAnimationFrame(drawVideoFrame)
        recorder.start()
        recorder.ondataavailable = function (e) {
            //saveChunks
            chunks.push(e.data)
        }
    }

    //處理停止的邏輯
    function stopRecord() {
        //停止影片串流的來源
        sourceTrack.stop()
        //停止Canvas錄制畫面
        recorder.stop()
        //停止請求requestAnimationFrame
        cancelAnimationFrame(rafId)
    }

    //取得串流失敗的錯誤處理
    function getUserMediaError(error) {
        console.log('navigator.webkitGetUserMedia() errot: ', error)
    }


    //按下開始錄制鈕
    dRecordBtn.onclick = function () {
        if (!isRecoding) {
            isRecoding = true
            dRecordBtn.style = "display:none"
            dStopBtn.style = ""

            //設定可以選擇媒體來源，以便開始處理串流
            let captureRequestID = chrome.desktopCapture.chooseDesktopMedia(["screen", "window", "tab"], function (streamId) {
                var audioConstraint = {
                    mandatory: {
                        chromeMediaSource: 'desktop',
                        chromeMediaSourceId: streamId
                    }
                }
                //使用 Navigator.getUserMedia拿到串流並開始處理
                navigator.mediaDevices.getUserMedia({
                    audio: audioConstraint,
                    video: {
                        mandatory: {
                            chromeMediaSource: 'desktop',
                            chromeMediaSourceId: streamId,
                            maxWidth: screen.width,
                            maxHeight: screen.height
                        }
                    }
                })
                    .then(record)
                    .catch(getUserMediaError)
                console.log(streamId)
            })
        }
    }
    //按下停止錄制鈕
    dStopBtn.onclick = function () {
        if (isRecoding) {
            //處理停止事件
            stopRecord()
            //處理UI
            isRecoding = false
            dRecordBtn.style = ""
            dStopBtn.style = "display:none"
        }
    }
})(window)
