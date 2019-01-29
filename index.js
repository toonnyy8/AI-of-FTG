const koa = require('koa');
const app = new koa();
const path = require('path');
const serve = require('koa-static');

const main = serve(path.join(__dirname + "/build/"));

app.use(main);
app.listen(8080);

var robot = require("robotjs");
