console.log('Hello from sw.js')

let CACHE_NAME = "cacheV1"
var urlsToCache = [
    '/',
    './main.fb6bbcaf.js',
    './agent.347b5559.js',
    './main.fb6bbcaf.js.map',
    './agent.347b5559.js.map'
]


self.addEventListener("install", (e) => {
    e.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('Opened cache')
                return cache.addAll(urlsToCache)
            })
    )
})

self.addEventListener('fetch', (e) => {
    e.respondWith(
        caches.match(e.request)
            .then((response) => {
                // Cache hit - return response
                if (response) {
                    return response
                }
                return fetch(e.request)
            })
    )
})