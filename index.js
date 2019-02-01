const koa = require('koa')
const path = require('path')
const serve = require('koa-static')
const Router = require('koa-router')
const koaBody = require('koa-body')
const app = new koa()
const router = new Router()

var robot = require("robotjs")

const main = serve(path.join(__dirname + "/build/"))

app.use(main)
app.listen(8080)
app.use(koaBody())

router.post('/output', async (ctx, next) => {
    console.log(ctx.request.body)
    let img = robot.screen.capture(0, 0, 1920, 1080)
    ctx.body = {
        "state": "ok der~",
        "screenshot": 1
    }
})
app.use(router.routes()).use(router.allowedMethods());

