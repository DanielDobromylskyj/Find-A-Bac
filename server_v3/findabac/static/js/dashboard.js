let loaded_task_ids = [];


function createTaskElement(taskName, progressPercent) {
  const task = document.createElement("div");
  task.className = "task";

  const content = document.createElement("div");
  content.className = "task-content";

  const name = document.createElement("div");
  name.className = "task-name";
  name.textContent = taskName;

  const progressWrapper = document.createElement("div");
  progressWrapper.className = "progress-wrapper";

  const progressBar = document.createElement("div");
  progressBar.className = "progress-bar";
  progressBar.style.width = progressPercent + "%";

  progressWrapper.appendChild(progressBar);

  const button = document.createElement("button");
  button.className = "open-btn";
  button.textContent = "Open";

  content.appendChild(name);
  content.appendChild(progressWrapper);
  content.appendChild(button);

  task.appendChild(content);

  return task;
}


function update_processing_queue() {
    fetch("/api/tasks")
        .then(response => response.json())
        .then(tasks => {
            const task_divs = document.getElementById("queue");
            Array.from(tasks).forEach(task => {
                if (!(loaded_task_ids.includes(task.task_id))) {
                    if (task.percentage >= 0) {
                        const html = createTaskElement(task.name, task.percentage);
                        task_divs.appendChild(html);

                        loaded_task_ids.push(task.task_id);

                        console.log("Added task", task);
                    }
                }

                Array.from(task_divs.children).forEach(element => {  // Borked
                    if (element.tagName === "DIV") {
                        if (element.getElementsByClassName("task-name")[0].innerHTML === task.name) {
                            const percentage_bar = element.getElementsByClassName("progress-bar")[0];
                            if (task.percentage === 0) {
                                percentage_bar.style.width = `100%`;
                                percentage_bar.classList.add("shiny");
                                percentage_bar.style.backgroundColor  = '#ffcc00';

                            } else if (task.percentage > 0) {
                                percentage_bar.style.width = `${task.percentage}%`;
                                percentage_bar.classList.remove("shiny");
                                percentage_bar.style.backgroundColor  = '#3478f6';
                            }

                        }
                    }
                })

            })
        })
}



setInterval(update_processing_queue, 1000);