hostCmd = null, hostCam = null ;
function setHost(host,port){ hostCmd = 'ws://'+host+':'+port+'/cmd'; hostCam = 'ws://'+host+':'+port+'/cam'; }

$(document).ready(function(){
    var socketCmd = null, socketCam = null;
    var updateTimers = false; // update or not timers data from camera
    var updateFrame = false; // update or not picture from camera
    var timeNow = 0; // value for fps calculation
    // var isSpinOn = false;
    var configCamServer = {} // server camera config
    var configCamLocal = {} // local camera config
    var initTimers = {'N':0,'fps':0, 'detect':0, 'location':0,'names':0,'encode':0};
    var configDisplyItems = ['cId',"cName","cUrl", "vdScl", "cConn", "frSkip"]
    // ,"faceBorder","scaleFactor", "compareValue", "isEncode", "isDetect", "isCompare","isCompareSlow", "classifierIndex", "imgWebFormat", "imgWebQuality", "confThreshold"]

    $('.toast').on('hidden.bs.toast', function () { console.log("hidden"); console.log($('#toastOne > #toastText')) })

    $('.dropdown-item-config').on('click', function (e) {e.preventDefault();
      console.log($(this))
      $('#'+e.target.name).text($(this).text());$('.dropdown-toggle').dropdown('toggle');
      configCamLocal[e.target.name] =  parseInt(e.target.id.charAt(0),10);
      compareCamConfigs(); e.stopPropagation();
    })

    $('.form-control-config').on('input', function(e){ e.preventDefault();
      // $('#'+e.target.id+"Out").val($(this).val());      
      console.log($(this).val())
      let tp = $(this).attr('type');
      if(tp=='checkbox'){configCamLocal[e.target.id] = $(this).prop("checked"); // if ( elem.checked ) // if ( $( elem ).prop( "checked" ) ) // if ( $( elem ).is( ":checked" ) )
      }else if(tp=='range'){configCamLocal[e.target.id] = parseFloat($(this).val());
      }else if(tp=='text'){ configCamLocal[e.target.id] = $(this).val();
      }else{ console.log('error!!!! input value')}
    // }else if(tp='button'){
    //   console.log('button')
    //   console.log($(this).text())
    //   configCamLocal[e.target.id] = $(this).text();
    // }
      compareCamConfigs();
      console.log(configCamLocal);
      console.log(configCamServer)
    })

    // $('.custom-switch').on('input', function(e){configCamLocal[e.target.id] = $('#'+e.target.id).val(); compareCamConfigs();})

    // $('#spinModalStop').on('click',function(e){$('#spinModalText').val(''); $('#spinModal').modal('hide'); isSpinOn = false; console.log("Stop modal");})

    $('#camStart').on('click', function(e) { e.preventDefault(); 
        if(socketCmd!=null){if(this.classList.contains('btn-primary')){// !!!!! Starting Camera
            // $('#spinModal').modal('show'); isSpinOn = true;
            // $('#spinModalText').val("Starting...\n");
            $(this)[0].classList.add('disabled');
            $(this).append($('<span id="confApplSpinner" class="spinner-border spinner-border-sm" role="status"></span>'));
            socketCmd.send(JSON.stringify({'0':'camStart','1':configCamServer['cId']}));
          }else{updateFrame = false; socketCmd.send(JSON.stringify({'0':'camStop','1':configCamServer['cId']})); }
        }else{alert("Lost server connection. Please reload page"); }
    });

    $('#camUpdateData').on('click', function(e) { e.preventDefault();
      if(this.classList.contains('disabled')){ 
        let str = "Please wait 2 sec"; if($('#camStart')[0].classList.contains('btn-primary')){str = "Please start camera before.";} alert(str);
      }else{ updateTimers = false; 
        if(this.classList.contains('btn-primary')){ console.log("Try start data.");
          if(socketCmd!=null){ btnStop($('#camUpdateData'),'Stop data');
            updateTimers = true; socketCmd.send(JSON.stringify({'0':'takeTimers','1':configCamServer['cId']}));
          }else{console.log("Cmd socket is unavailable.")}
        }else{console.log("Stop data."); btnStart($('#camUpdateData'),'Show data');
      }}
    });

    $('#startStream').on('click', function(e) { e.preventDefault(); 
      showLog("App","Camera stream starting...");
      if(this.classList.contains('disabled')){
        let str = "Please wait 2 sec"; if($('#camStart')[0].classList.contains('btn-primary')){str = "Please start camera before.";} alert(str);
      }else{if(this.classList.contains('btn-primary')){startCam();}else{ socketCam.close(); 
      }}
    });

    $('#camConfigSave').on('click', function(e) {e.preventDefault(); console.log("configSave");
    showLog("App","Camera config saving...");
      if(socketCmd!=null){socketCmd.send(JSON.stringify({'0':'camConfSave','1':configCamLocal['cId'],'2':configCamLocal}));}
      // console.log(configCamLocal)
    });

    $('#camConfigUndo').on('click', function(e) {e.preventDefault();  console.log("configUndo");    
        if(compareCamConfigs()){ showLog("App","Camera configs are similar");
        }else{ configCamLocal = Object.assign({}, configCamServer);  displayCamConfigs(configCamServer); showLog("App","Camera config undo");}
    });

    $('#camConfigApply').on('click', function(e) {e.preventDefault();
        if(compareCamConfigs()){showLog("App","Camera configs are similar");
          }else{ showLog("App","Camera config appling...");
            if(socketCmd!=null){
              socketCmd.send(JSON.stringify({'0':'camConfApply','1':configCamServer['cId'],'2':configCamLocal}));
              $(this).append($('<span id="confApplSpinner" class="spinner-border spinner-border-sm" role="status"></span>'));
            }else{showLog("App","Please reload page");}
          }
      });

    if (!("WebSocket" in window)) {alert("Your browser does not support web sockets");
    }else{
    // !!!!!!!!!!!!! main start script
    // console.log($('#isDetect')[0])
      console.log((new Date).toLocaleTimeString()+' start main script');
      socketCmd = new WebSocket(hostCmd);
        if(socketCmd){
          socketCmd.onopen = function(){showLog("Server:","Started"); $('#camStart')[0].classList.remove('disabled'); }

          socketCmd.onmessage = function(msg){
                // console.log("server says: "+msg.data);
                data =  JSON.parse(msg.data)
                switch (data[0]) {
                    case 'timers': updateData(data[1]);
                        if(updateTimers){socketCmd.send(JSON.stringify({'0':'takeTimers','1':configCamServer['cId']}));
                        }else{showLog("App","cmd updateTimers is stoped");updateData(initTimers);} break;
                    case 'log': showLog(data[2],data[1]); break;
                    case 'system':
                        // console.log("server: "+msg.data);
                        switch (data[1]) {
                          case 'cameraStopOk':
                              if(socketCam){socketCam.close();}
                              updateTimers = false;  updateFrame = false;  configCamServer = {}; 
                              displayCamConfigs(configCamServer); configCamLocal = {}; updateData(initTimers);
                              $('#camUpdateData')[0].classList.add('disabled');
                              // btnStart($('#camUpdateData'));
                              $('#startStream')[0].classList.add('disabled');
                              btnStart($('#camUpdateData'),'Show data');
                              $('#camStart')[0].classList.remove('disabled');
                              $('#camConfigApply').children('#confApplSpinner').remove();
                              btnStart($('#camStart'),'Connect');
                              showLog(data[2],"Camera stoped"); break; 
                          case 'confApplied':
                              configCamServer =  data[3];
                              configCamLocal = Object.assign({}, data[3]);
                              displayCamConfigs(configCamServer);
                              $('#camConfigApply').children('#confApplSpinner').remove();
                              showLog(data[2],"Camera settings updated"); break;
                          case 'takeConfig': showLog(data[2],"Loading new camera settings");
                            setTimeout(function(){if(socketCmd){socketCmd.send(JSON.stringify({'0':'camConfTake','1':'onRestart','2':data[2]}))}}, 200);  break;
                          default:  showLog("App","Unknown system server command"); break; 
                        }
                        break;
                    case 'config':
                        // console.log("server config: "+data[1]);
                        if(typeof(data[1]['cId']) !== 'undefined'){
                          configCamServer = data[1];
                          configCamLocal = Object.assign({}, data[1]);
                          $('#camUpdateData')[0].classList.remove('disabled');
                          btnStart($('#camUpdateData'),'Show data');
                          $('#startStream')[0].classList.remove('disabled');
                          $('#camStart')[0].classList.remove('disabled'); $('#camConfigApply').children('#confApplSpinner').remove();
                          btnStart($('#startStream'),'Show picture');
                          btnStop($('#camStart'),'Disconnect');
                          displayCamConfigs(configCamServer);
                          showLog(data[2],"Camera config updated"); 
                        }else{ showLog(data[2],"Camera config error"); } break;
                    case 'error': 
                      // console.log("server: "+msg.data);
                      showLog(data[2],data[1]);
                      $('#camStart')[0].classList.remove('disabled'); $('#camConfigApply').children('#confApplSpinner').remove();
                      switch (data[1]) {
                        case 'cameraClosed':
                          updateTimers = false;  updateFrame = false;  configCamServer = {}; displayCamConfigs(configCamServer); configCamLocal = {}; updateData(initTimers);
                          $('#startStream')[0].classList.add('disabled'); $('#camUpdateData')[0].classList.add('disabled');
                          $('#camConfigApply').children('#confApplSpinner').remove();
                          
                          updateData(initTimers); btnStart($('#camUpdateData'),'Show data'); btnStart($('#startStream'),'Show picture'); btnStart($('#camStart'),'Connect');
                          break;
                        case 'unKnowCommand':  showLog(data[2],"Unknown client command"); break;
                        default:  showLog(data[2],"Unknown error command");} break;
                    default: showLog(data[2],"Unknown server command");
                }
            }

          socketCmd.onclose = function(){
            // if(isSpinOn){$('#spinModal').modal('hide'); isSpinOn = false;}
            $('#camStart')[0].classList.add('disabled');
            $('#startStream')[0].classList.add('disabled');
            $('#camUpdateData')[0].classList.add('disabled');
            btnStart($('#camUpdateData'),'Show data'); btnStart($('#startStream'),'Show picture'); btnStart($('#camStart'),'Connect');
            updateTimers = false;  updateFrame = false;  configCamServer = {}; displayCamConfigs(configCamServer); configCamLocal = {}; updateData(initTimers);
            showLog('App',"Camera connection closed"); }
          }else{showLog('App',"Invalid camera connection socket");}
        }

      function compareCamConfigs(){ let tr = true;
        if(Object.keys(configCamServer).length){
          if(JSON.stringify(configCamLocal)==JSON.stringify(configCamServer)){ 
            $('#camConfigApply')[0].classList.add("disabled");
            $('#camConfigUndo')[0].classList.add("disabled");
            $('#camConfigSave')[0].classList.add("disabled");
            }else{ 
              $('#camConfigApply')[0].classList.remove("disabled"); 
              $('#camConfigUndo')[0].classList.remove("disabled"); 
              $('#camConfigSave')[0].classList.remove("disabled"); 
              tr=false}}
          return tr;
        }

      function displayCamConfigs(data){
        if(data['cId']){$('#cameraIdHeaderLabel').text(data['cId'])}else{$('#cameraIdHeaderLabel').text('None')}
        configDisplyItems.forEach(function(item){let pointer = $('#'+item); // console.log(data[item])
          if(pointer){if(pointer[0]){
                  if(pointer[0]['type']){ let tp = pointer.attr('type'), isKey = false;
                    if(!(item in data)){data[item]=''; isKey=true;}
                    if('text'==tp){pointer.val(data[item]);
                    } else if ('range'==tp) {pointer.val(data[item]); $('#'+item+'Out').val(data[item]);
                    } else if ('checkbox'==tp){ pointer.prop('checked', data[item]); 
                    } else if ('button'==tp){if(isKey){pointer.text("None");}else{$('#'+data[item]+item).click(); $('.dropdown-toggle').dropdown('toggle');}
                    }// else {console.log("item type is not detected"); console.log($('#'+item));}
                  }//else{console.log("pointer type === undefined");}
                  }//else{console.log("pointer === undefined");}
                }//else{console.log("obj undefined "+item);}
          }); $(window).scrollTop(0);}
      
      function updateData(data){
        $('#N').text(data['N']);
        $('#serverFps').text(data['fps']);
        $('#detect').text(data['detect']);
        $('#location').text(data['location']);
        $('#names').text(data['names']);
        $('#encode').text(data['encode']);
      }

      function startCam(){
          socketCam = new WebSocket(hostCam);
          if(socketCam){
            socketCam.onopen = function(){
              updateFrame = true;
              timeNow = Date.now();
              $('#startStream')[0].classList.remove('disabled');
              btnStop($('#startStream'),'Stop picture');
              showLog('App',"Camera stream connection started");
              socketCam.send(1);
            }

            socketCam.onmessage = function(msg){
              if(updateFrame){if(msg.data!=null){ $("#camImg").attr('src', URL.createObjectURL(msg.data));}
              $('#localFps').text(Math.round(1000/(Date.now() - timeNow))); timeNow = Date.now();
              setTimeout(function(){if(updateFrame){if(socketCam){socketCam.send(1);}}}, 30);
              }else{(console.log("skip empty frame"));}
            }

            socketCam.onclose = function(){
              updateFrame = false; 
              btnStart($('#startStream'),"Show picture");
              showLog('App',"Camera stream connection closed");
            }
          }else{console.log("Invalid camera stream connection socket");}	
        }

        function btnStop(btn,label){if(btn[0]){btn[0].classList.replace('btn-primary','btn-danger'); btn.text(label);}}
        function btnStart(btn,label){if(btn[0]){btn[0].classList.replace('btn-danger','btn-primary'); btn.text(label);}} //"Start"
        function showLog(header,msg){elem = $("#toastOne").clone().appendTo($('#toastsArea')).toast('show'); elem.children('#toastText').text(msg); elem.find('#toastHead').text(header); console.log((new Date()).toLocaleTimeString()+' Log>',header,msg)}
});


