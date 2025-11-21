# Fantasy Stats

This repo was designed to justify that PF - PA is a the optimal metric to be correlated with winning percentage in fantasy football leagues.

### Results Summary

Look in [plots](plots) for visualizations of the results.

| Metric        | $R^2$ Correlation with Win % |
|---------------|----------------------------- |
| PF            | 0.2290                       |
| PA            | 0.0799                       |
| PF - PA       | 0.8144                       |


### How to gather data:

Go to 
`https://fantasy.espn.com/football/league/standings?leagueId=<your_id>&seasonId=<your_season>`
Then open the browser console (F12) and paste in the following code:

```javascript
let table = document.querySelectorAll(".season--stats--table .Table__Scroller .Table__TBODY")
let rows = table[0].children
let acc_str = ""
for (let i = 0; i < rows.length; i++) {
    let cur_row = rows[i].children;
    let pf = cur_row[0].textContent;
    let pa = cur_row[1].textContent;
    let wl_raw = cur_row[2].textContent;
    let wl_parts = wl_raw.split("-");
    let wl_perc = Number(wl_parts[0])/(Number(wl_parts[0]) + Number(wl_parts[1]));
    acc_str += `${pf}, ${pa}, ${wl_perc}\n`;
}
console.log(acc_str)
```

Then copy this output and paste it into a CSV file.

### How to run:

```bash
uv run main.py --csv <path_to_your_csv>
```