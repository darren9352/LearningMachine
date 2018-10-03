$(document).ready(function() {

  var dropContainer = document.getElementById('drop-container');
  dropContainer.ondragover = dropContainer.ondragend = function() {
    return false;
  };

  dropContainer.ondrop = function(e) {
    e.preventDefault();
    loadImage(e.dataTransfer.files[0])
  }

  $("#browse-button").change(function() {
    loadImage($("#browse-button").prop("files")[0]);
  });

  $('.modal').modal({
    dismissible: false,
    ready: function(modal, trigger) {
      $.ajax({
        type: "POST",
        url: '/classify_image/classify/api/',
        data: {
          'image64': $('#img-card').attr('src'),
          'attack': $("#attack_algorithm option:selected").text(),
          'iterate': $('#iterate').val(),
          'model': $('input:radio[name=selected_model]:checked').val(),
          'sample':$("#mnist_number option:selected").text(),
          'target':$("#target_number option:selected").text(),
          'mnist_algorithm':$("#mnist_attack_algorithm option:selected").text(),
        },
        dataType: 'text',
        success: function(data) {
          loadStats(data)
        }
      }).always(function() {
        modal.modal('close');
      });
    }
  });

  $('#go-back, #go-start').click(function() {
    $('#img-card').removeAttr("src");
    $('#mnist-img-card').removeAttr("src");
    $('#adver-card').removeAttr("src");
    $('#adver-card').removeAttr("data-gif");
    $('#adver-card').removeAttr("width");
    $('#adver-card').removeAttr("height");
    $('#stat-table').html('');
    switchAdver(0);
    switchCard(0);
    checked = $('input:radio[name=selected_model]:checked').val();
    if(checked == 'mnist'){
      $('.mnist-pure-container').animate({
        opacity: 0
      }, {
        duration: 200,
        queue: false,
      }).css("z-index", 0);
      switchModel(1);
    }
    document.getElementById('result-card').style.display = 'none';
  });

  $('#upload-button').click(function() {
    $('.modal').modal('open');
  });
  $('#mnist-upload-button').click(function() {
    $('.modal').modal('open');
  });
});

switchInput = function(cardNo) {
  var containers = [".dd-container", ".mnist-uf-container"];
  var visibleContainer = containers[cardNo];
  for (var i = 0; i < containers.length; i++) {
    var oz = (containers[i] === visibleContainer) ? '1' : '0';

    $(containers[i]).animate({
      opacity: oz
    }, {
      duration: 200,
      queue: false,
    }).css("z-index", oz);
  }
}
switchModel = function(cardNo) {
  var containers = [".dd-container", ".mnist-dd-container"];
  var visibleContainer = containers[cardNo];
  for (var i = 0; i < containers.length; i++) {
    var oz = (containers[i] === visibleContainer) ? '1' : '0';

    $(containers[i]).animate({
      opacity: oz
    }, {
      duration: 200,
      queue: false,
    }).css("z-index", oz);
  }
}
switchAdver = function(cardNo) {
  var containers = [".ad-temp-container", ".ad-container"];
  var visibleContainer = containers[cardNo];
  for (var i = 0; i < containers.length; i++) {
    var oz = (containers[i] === visibleContainer) ? '1' : '0';

    $(containers[i]).animate({
      opacity: oz
    }, {
      duration: 200,
      queue: false,
    }).css("z-index", oz);
  }
}
switchCard = function(cardNo) {
  var containers = [".dd-container", ".uf-container", ".dt-container"];
  var visibleContainer = containers[cardNo];
  for (var i = 0; i < containers.length; i++) {
    var oz = (containers[i] === visibleContainer) ? '1' : '0';

    $(containers[i]).animate({
      opacity: oz
    }, {
      duration: 200,
      queue: false,
    }).css("z-index", oz);
  }
}

showStat = function() {
    $('.dt-container').animate({
        opacity: 1
    },{
        duration:200,
        queue: false,
    }).css("z-index",1);
}

loadImage = function(file) {
  var reader = new FileReader();
  reader.onload = function(event) {
    $('#img-card').attr('src', event.target.result);
  }
  reader.readAsDataURL(file);

  switchCard(1);
}

loadStats = function(jsonData) {
  showStat();
  $('.uf-button').animate({
      opacity: 0
    }, {
      duration:200,
      queue: false,
    }).css("z-index", 0);
  //switchCard(2);
  var data = JSON.parse(jsonData);
  var chartData = [{x:[], y:[], type:'bar', text: [], textposition: 'auto'}];
  var adverData = [{x:[], y:[], type:'bar', text: [], textposition: 'auto'}];

  if (data["success"] == true) {
    
    $('#adver-card').attr('src', data["adverimage"]);
    $('#adver-card').attr('data-gif', data["adverimage_gif"]);

    $('#adver-card').attr('object-fit', 'cover');
    $('#adver-card').attr('object-position', 'centor');
    $('#adver-card').attr('width', 'inherit');
    $('#adver-card').attr('height', '100%');
    
   
    //$('#adver-card').attr('src', data["adverimage_gif"]);

    if (data["model"] == 'mnist') {
      $('#mnist-img-card').attr('src', data["input_image"]);
      switchInput(1);
    }
    switchAdver(1);
    var original;
    var adversarial;
    var original_max = -1;
    var adversarial_max = -1;
	var attack_speed = data["attack_speed"];
	attack_speed = attack_speed.toFixed(5);
    for (category in data['confidence']) {
      var percent = Math.round(parseFloat(data["confidence"][category]) * 100);
        if(original_max < percent){
          original_max = percent;
          original = category;
        }
        chartData[0].x.push(category);
        chartData[0].y.push(percent);
		    chartData[0].text.push(percent);
    }
    for (category in data['adversarial']) {
      var percent = Math.round(parseFloat(data["adversarial"][category]) * 100);
        if(adversarial_max < percent){
          adversarial_max = percent;
          adversarial = category;
        }
        adverData[0].x.push(category);
        adverData[0].y.push(percent);
		    adverData[0].text.push(percent);
    }

	origin_layout = {
		title: 'Original',
	};

	adver_layout = {
		title: 'Adversarial',
	};

    Plotly.newPlot('stat-table', chartData, origin_layout);
    Plotly.newPlot('adver-table', adverData, adver_layout);
    document.getElementById('original').innerHTML = original;
    document.getElementById('adversarial').innerHTML = adversarial;
	  document.getElementById('attack_speed').innerHTML = attack_speed;
    document.getElementById('result-card').style.display = 'block';
  }

}
