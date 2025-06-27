let loaded_task_ids = [];
let loaded_task_ids_recent = [];


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
function createRecentTaskElement(taskName) {
  const task = document.createElement("div");
  task.className = "task";

  const content = document.createElement("div");
  content.className = "task-content-completed";

  const name = document.createElement("div");
  name.className = "task-name";
  name.textContent = taskName;

  const button = document.createElement("button");
  button.className = "open-btn-completed";
  button.textContent = "Open";

  content.appendChild(name);
  content.appendChild(button);

  task.appendChild(content);

  return task;
}

function update_processing_queue() {
    fetch("/api/tasks")
        .then(response => response.json())
        .then(tasks_raw => {
            const tasks = Array.from(tasks_raw);
            const task_divs = document.getElementById("queue");

            const all_task_names = tasks.map(task => task.name);

            tasks.forEach(task => {
                if (!(loaded_task_ids.includes(task.task_id))) {
                    if (task.percentage >= 0) {
                        const html = createTaskElement(task.name, task.percentage);
                        task_divs.appendChild(html);

                        loaded_task_ids.push(task.task_id);

                        console.log("Added task", task);
                    }
                }

                Array.from(task_divs.children).forEach(element => {
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

            Array.from(task_divs.children).forEach(element => {
                if (element.tagName === "DIV") {
                    if (!(all_task_names.includes(element.getElementsByClassName("task-name")[0].innerHTML))) {
                        element.remove();
                    }
                }
            })
        })

     fetch("/api/archived/recent")
        .then(response => response.json())
        .then(tasks => {
            const task_divs = document.getElementById("completed-section");
            Array.from(tasks).forEach(task => {
                if (!(loaded_task_ids_recent.includes(task.task_id))) {
                    const html = createRecentTaskElement(task.name);
                    task_divs.appendChild(html);

                    loaded_task_ids_recent.push(task.task_id);
                }
            })
        })
}



setInterval(update_processing_queue, 1000);