'use strict';
function parseCanonicalCSV(text,features){
  const rows=text.trim().split(/\r?\n/).map(row=>row.split(',').map(cell=>cell.trim()));
  if(JSON.stringify(rows.shift())!==JSON.stringify(['time_s',...features])||rows.length!==15)
    throw Error('Use the exact canonical header and 15 data rows.');
  if(rows.some(row=>row.length!==11||row.some(cell=>cell==='')))
    throw Error('Missing values are not allowed; provide all 11 numeric columns.');
  const numeric=rows.map(row=>row.map(Number));
  if(numeric.some(row=>row.some(value=>!Number.isFinite(value))))
    throw Error('All sensor values and timestamps must be finite numbers.');
  return {raw:numeric.map(row=>row.slice(1)),timestamps:numeric.map(row=>row[0])};
}
if(typeof module!=='undefined')module.exports={parseCanonicalCSV};
