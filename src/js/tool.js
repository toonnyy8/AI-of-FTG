export class Loop {
    constructor(func, period) {
        this.period = period
        this.count = 2
        this.func = func
    }

    run(...args) {
        if (this.count == 1) {
            this.func(...args)
        }
        this.count = this.count < this.period ? this.count + 1 : 1

    }
}