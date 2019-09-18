$(document).ready(function(){

    $(document).on("click", "a.list-group-item" , function() {
        data = lines[$(this).attr('id')]
        // console.log(Object.keys(data).length)
        if(Object.keys(data).length>3){
            // console.log(data);
            $('#imgModal').modal('show');
            $('#imgModalBody').append(data[3],data[0],data[2],data[4]);
        }
    });

 var lines=[];
    function getLog(name,counter){
        fetch("http://localhost:9090/getlog?name="+name+'&count='+counter, {
                method : "GET",
                mode: "cors", // no-cors, cors, *same-origin
                cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
                credentials: "same-origin", // include, *same-origin, omit
                headers: {"Content-Type": "text/plain"},
                //redirect: "/", // manual, *follow, error
                referrer: "no-referrer", // no-referrer, *client
                // body: JSON.stringify(answer), // body data type must match "Content-Type" header
              }).
              then(response => response.text()
                ).then((text) => {
                    arrText = text.split('\n');
                    arrText.forEach(function(item,index){
                        lineArr =item.split('##');
                        lines.push(lineArr);
                        if(lineArr[1]){
                        if(lineArr[1]=='INFO'){lineArr[1]='light'}
                        // arrLinks.push(lineArr[3])
                        $("#eventsList").append('<a class="list-group-item list-group-item-action list-group-item-'+lineArr[1]+'" href="#list-home" id="'+index+'" >'+lineArr[0]+'  '+lineArr[2]+'</a>');
                    }
                    })
            }).catch(error => {
              console.error('Error:', error);
            });
    }
    getLog('camera_USB0.log',10);
    // console.log(lines);
});