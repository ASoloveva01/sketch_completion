var canvas;
var context;
var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var paint = false;
var curColor = "#000000";

/**
    - Подготовка холста: Основные функции
**/
function drawCanvas() {

    canvas = document.getElementById('canvas');
    context = document.getElementById('canvas').getContext("2d");

    $('#canvas').mousedown(function (e) {
        var mouseX = e.pageX - this.offsetLeft;
        var mouseY = e.pageY - this.offsetTop;

        paint = true;
        addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
        redraw();
    });

    $('#canvas').mousemove(function (e) {
        if (paint) {
            addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
            redraw();
        }
    });

    $('#canvas').mouseup(function (e) {
        paint = false;
    });
}

/**
    - Созраняет позицию клика
**/
function addClick(x, y, dragging) {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
}

/**
    - Очищает холст и перерисовывает
**/
function redraw() {

    context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
    context.strokeStyle = curColor;
    context.lineJoin = "round";
    context.lineWidth = 3;
for (var i = 0; i < clickX.length; i++) {
    context.beginPath();
    if (clickDrag[i] && i) {
        context.moveTo(clickX[i - 1], clickY[i - 1]);
    } else {
        context.moveTo(clickX[i] - 1, clickY[i]);
    }
    context.lineTo(clickX[i], clickY[i]);
    context.closePath();
    context.stroke();
}
}

/**
    - Очищает холст и всю историю рисования
    - Кодирует изображение
    - Отправляет закодированное изображение на сервер через ajax
    - Ответ сервера присваивает image.src для показа изображения
**/

function post_image() {
            context.clearRect(0, 0, context.canvas.width, context.canvas.height);
            clickX = new Array();
            clickY = new Array();
            data = canvas.toDataURL();
            $.ajax({
                type: "POST",
                url: "/predict",
                data: JSON.stringify({"init_img": data}),
                success: function(response) {
                    var json = jQuery.parseJSON(response);
                    var image = document.getElementById('img');
                    image.src = "data:image/png;base64," + json.gen_img;

                },
                error: function(error) {
                    alert(error);
                },
                contentType: "application/json"
             });

}