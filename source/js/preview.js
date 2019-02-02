import * as gpuJs from "gpu.js"

global.tempBuffer = {};

(async function (exports) {
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
    var videoWidth = 192 //* 3
    var videoHeight = 108 //* 3
    video.autoplay = true
    video.width = videoWidth
    video.height = videoHeight

    //畫面展示需要，使用jquery dom 
    var dRecordBtn = document.getElementById("recodBtn")
    var dStopBtn = document.getElementById("stopBtn")
    dStopBtn.style = "display:none"

    let preFrame

    let originCanvas = document.createElement("canvas")
    document.getElementById("view").appendChild(originCanvas)
    //錄制畫面用的canvas
    originCanvas.width = videoWidth
    originCanvas.height = videoHeight
    originCanvas.style = "width:800px"
    const ctx = originCanvas.getContext('2d')
    ctx.fillRect(0, 0, videoWidth, videoHeight)

    let img = new Image()
    img.src = originCanvas.toDataURL()

    console.log(ctx.getImageData(0, 0, videoWidth, videoHeight))

    let greyCanvas = document.createElement("canvas")
    document.getElementById("view").appendChild(document.createElement("br"))
    document.getElementById("view").appendChild(greyCanvas)
    //轉為灰階用的canvas
    greyCanvas.width = videoWidth
    greyCanvas.height = videoHeight

    const gl = greyCanvas.getContext('webgl2', {})

    console.log(gl)

    let gpu = new GPU({
        "mode": "gpu",
        greyCanvas,
        "webGl": gl
    })

    kernel = function (data) {
        var x = this.thread.x,
            y = this.thread.y

        var n = 4 * (x + this.constants.w * (this.constants.h - y));
        var grey = .399 * data[n] + .587 * data[n + 1] + .114 * data[n + 2]
        this.color(grey / 255, grey / 255, grey / 255, 1);
    }

    const toGray = gpu.createKernel(
        function (data) {
            var x = this.thread.x,
                y = this.thread.y

            var n = 4 * (x + this.constants.w * (this.constants.h - y));
            var grey = .399 * data[n] + .587 * data[n + 1] + .114 * data[n + 2]

            this.color(grey / 255, grey / 255, grey / 255, 1);
        }
    )
        .setConstants({ w: videoWidth, h: videoHeight })
        .setOutput([videoWidth, videoHeight])
        .setGraphical(true)
        .setDebug(true)



    /*const toGray = function (a) {
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
 
    }).setOutput([videoWidth, videoHeight])*/
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

        function drawVideoFrame(time) {
            rafId = requestAnimationFrame(drawVideoFrame)

            toGray(new Float32Array(ctx.getImageData(0, 0, videoWidth, videoHeight).data))

            ctx.drawImage(video, 0, 0, videoWidth, videoHeight)
        }
        //開始截取畫面並把requestAnimationFrame的id儲存起來以便控制

        cStream = originCanvas.captureStream(60)
        recorder = new MediaRecorder(cStream)

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
