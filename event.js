chrome.browserAction.onClicked.addListener(function () {
    chrome.tabs.create({
        url: "/build/index.html"
    }, function (tab) {
        console.log('window open');
    });
});

